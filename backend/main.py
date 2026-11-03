from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import ee

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicjalizacja GEE
try:
    ee.Initialize(project='uhi-project-inzynierka')
    print("Serwer połączony z Google Earth Engine!")
except Exception as e:
    print("Błąd inicjalizacji GEE:", e)

@app.get("/")
def read_root():
    return {"message": "Witaj w API WebGIS UHI Project!"}

# --- KONFIGURACJA MIAST ---
# "adm2_name" to nazwa w kolekcji FAO/GAUL/2015/level2 (pole ADM2_NAME).
# Jeśli dla któregoś miasta geometria wyjdzie pusta (błąd 404 z get_city_geometry),
# sprawdź dokładną nazwę w tej kolekcji (filtr ADM1_NAME == 'Poland') i popraw wpis poniżej.
CITIES = {
    "lublin": {"label": "Lublin", "adm2_name": "Lublin"},
    "warszawa": {"label": "Warszawa", "adm2_name": "Warszawa"},
    "chelm": {"label": "Chełm", "adm2_name": "Chelm"},  # GAUL zapisuje nazwy bez polskich znaków diakrytycznych
}

# --- FUNKCJE POMOCNICZE ---

def get_city_geometry(city_key: str):
    """Granica administracyjna miasta (FAO/GAUL level2, dopasowanie po ADM2_NAME)."""
    if city_key not in CITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Nieznane miasto '{city_key}'. Dostępne: {', '.join(CITIES.keys())}."
        )

    adm2_name = CITIES[city_key]["adm2_name"]
    admin_boundaries = ee.FeatureCollection("FAO/GAUL/2015/level2")
    matches = admin_boundaries.filter(ee.Filter.eq('ADM2_NAME', adm2_name))

    if matches.size().getInfo() == 0:
        raise HTTPException(
            status_code=404,
            detail=(f"Nie znaleziono granic administracyjnych dla '{adm2_name}' w GAUL. "
                     "Sprawdź dokładną wartość ADM2_NAME dla tego miasta.")
        )
    return matches.geometry()


def get_reference_ring(city_geom, buffer_km: float = 18, exclude_km: float = 2):
    """
    Geometria pierścienia referencyjnego (wiejskiego) wokół miasta:
    bufor od exclude_km do buffer_km od granicy miasta (sam pierścień, bez środka).
    """
    outer = city_geom.buffer(buffer_km * 1000)
    inner = city_geom.buffer(exclude_km * 1000)
    # 30 m zamiast 1 m - precyzja i tak jest ograniczona rozdzielczością pikseli Landsata,
    # a zbyt ciasny margines błędu przy operacjach na dużych buforach generuje
    # nadmiernie złożone geometrie (wolne/zawodne przy .getInfo()).
    return outer.difference(inner, ee.ErrorMargin(30))


def get_rural_mask():
    """
    Maska terenu 'wiejskiego' wg ESA WorldCover (10 m): zostają tylko piksele
    trawiaste (30) i rolnicze (40) — bez zabudowy, lasu, wody i innych klas,
    żeby pierścień referencyjny nie łapał przypadkowo innych miejscowości.
    """
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
    return worldcover.eq(30).Or(worldcover.eq(40))


def mask_landsat_clouds(image):
    """Maskuje chmury, cienie chmur i 'dilated cloud' na podstawie QA_PIXEL (Landsat C2 L2)."""
    qa = image.select('QA_PIXEL')
    dilated_cloud = 1 << 1
    cirrus = 1 << 2
    cloud = 1 << 3
    cloud_shadow = 1 << 4
    mask = (qa.bitwiseAnd(dilated_cloud).eq(0)
            .And(qa.bitwiseAnd(cirrus).eq(0))
            .And(qa.bitwiseAnd(cloud).eq(0))
            .And(qa.bitwiseAnd(cloud_shadow).eq(0)))
    return image.updateMask(mask)


def get_landsat_composite(roi, year: int, start_month: int = 6, end_month: int = 9):
    """
    Kompozyt medianowy z sezonu letniego (domyślnie czerwiec-wrzesień, exclusive end_month):
    każdy piksel to mediana ze wszystkich bezchmurnych obserwacji w oknie,
    zamiast pojedynczego 'najlepszego' zdjęcia — stabilniejsze i porównywalne
    między latami oraz między miastami.
    Zwraca None, jeśli w oknie nie ma żadnych zdjęć pokrywających roi.
    """
    start = f'{year}-{start_month:02d}-01'
    end = f'{year}-{end_month:02d}-01'

    collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .filterBounds(roi)
                  .filterDate(start, end)
                  .map(mask_landsat_clouds))

    if collection.size().getInfo() == 0:
        return None

    return collection.median()


def compute_lst_celsius(image):
    return image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)


def reduce_mean(image, geometry, scale=30):
    """Pomocnicza: średnia z danego pasma ST_B10 nad geometrią, z bezpiecznymi limitami."""
    value = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=scale,
        bestEffort=True,
        maxPixels=1e9
    ).get('ST_B10')
    return value


# --- ENDPOINTY ---

@app.get("/api/cities")
def list_cities():
    return {key: val["label"] for key, val in CITIES.items()}


# 1. Punktowy pomiar (z wyborem roku)
@app.get("/api/lst")
def get_land_surface_temperature(lat: float, lon: float, year: int = 2023, smooth: bool = True):
    try:
        point = ee.Geometry.Point([lon, lat])
        composite = get_landsat_composite(point, year)

        if composite is None:
            raise HTTPException(status_code=404, detail=f"Brak bezchmurnych danych dla roku {year} w tym miejscu.")

        # smooth musi być zgodne z tym, którą warstwę (wygładzoną czy surową) użytkownik
        # aktualnie widzi na mapie - inaczej popup znowu nie zgadzałby się z kolorem pod kursorem.
        lst_celsius = compute_lst_celsius(composite)
        if smooth:
            lst_celsius = lst_celsius.resample('bicubic')
        temperature = reduce_mean(lst_celsius, point).getInfo()

        if temperature is None:
            raise HTTPException(status_code=404, detail="Brak danych (piksel stale zachmurzony w tym oknie).")

        return {
            "latitude": lat,
            "longitude": lon,
            "year": year,
            "lst_celsius": round(temperature, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 2. Mapa LST (z wyborem miasta i roku)
# smooth=True (domyślnie) - wygładzona wersja (bicubic) do normalnego oglądania.
# smooth=False - surowe piksele 30m (widać siatkę/"kratę" rzeczywistej rozdzielczości
# Landsata) - do podglądu "co tak naprawdę zmierzył satelita" bez interpolacji.
@app.get("/api/map/lst")
def get_lst_map_url(city: str = "lublin", year: int = 2023, smooth: bool = True):
    try:
        roi = get_city_geometry(city)
        composite = get_landsat_composite(roi, year)

        if composite is None:
            raise HTTPException(status_code=404, detail=f"Brak bezchmurnych danych dla roku {year}.")

        lst_celsius = compute_lst_celsius(composite)
        if smooth:
            lst_celsius = lst_celsius.resample('bicubic')
        lst_clipped = lst_celsius.clip(roi)

        # Dynamiczny zakres skali kolorów: rozciągnięty na faktyczny rozkład temperatur
        # w TYM mieście i roku (percentyle 2-98, żeby pojedyncze piksele-odstające nie
        # psuły skali), zamiast sztywnego 20-45°C. Dzięki temu różnice przestrzenne
        # w obrębie miasta są widoczne nawet gdy całe miasto mieści się w wąskim
        # przedziale temperatur - kosztem tego, że ten sam kolor może oznaczać inną
        # temperaturę na różnych mapach (porównywanie "na oko" między latami/miastami
        # robimy liczbami ze statystyk/UHI, nie kolorem). Same wyliczone wartości LST
        # (stats, UHI) są od tej zmiany całkowicie niezależne - to tylko wizualizacja.
        percentiles = lst_clipped.reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]),
            geometry=roi,
            scale=30,
            bestEffort=True,
            maxPixels=1e9
        ).getInfo()

        vis_min = percentiles.get('ST_B10_p2')
        vis_max = percentiles.get('ST_B10_p98')
        if vis_min is None or vis_max is None:
            vis_min, vis_max = 20, 45
        elif vis_max - vis_min < 2:
            vis_min, vis_max = vis_min - 1, vis_max + 1

        vis_params = {
            'min': vis_min, 'max': vis_max,
            'palette': ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
        }

        map_id_dict = lst_clipped.getMapId(vis_params)

        return {
            "url": map_id_dict['tile_fetcher'].url_format,
            "attribution": f"Landsat 8 ({year}, mediana VI-IX, {'wygładzone' if smooth else 'surowe piksele 30m'}) / GEE",
            "vis_min": round(vis_min, 1),
            "vis_max": round(vis_max, 1)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 3. Mapa NDVI (z wyborem miasta i roku)
@app.get("/api/map/ndvi")
def get_ndvi_map_url(city: str = "lublin", year: int = 2023):
    try:
        roi = get_city_geometry(city)
        composite = get_landsat_composite(roi, year)

        if composite is None:
            raise HTTPException(status_code=404, detail=f"Brak bezchmurnych danych dla roku {year}.")

        ndvi = composite.normalizedDifference(['SR_B5', 'SR_B4'])
        ndvi_clipped = ndvi.clip(roi)

        vis_params = {
            'min': 0, 'max': 0.8,
            'palette': ['white', 'green']
        }

        map_id_dict = ndvi_clipped.getMapId(vis_params)

        return {
            "url": map_id_dict['tile_fetcher'].url_format,
            "attribution": f"Landsat 8 NDVI ({year}, mediana VI-IX) / GEE"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 4. Statystyki miasta (z wyborem miasta i roku)
@app.get("/api/stats")
def get_stats(city: str = "lublin", year: int = 2023):
    try:
        roi = get_city_geometry(city)
        composite = get_landsat_composite(roi, year)

        if composite is None:
            raise HTTPException(status_code=404, detail="Brak danych.")

        lst_celsius = compute_lst_celsius(composite)

        stats = lst_celsius.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.minMax(),
                sharedInputs=True
            ),
            geometry=roi,
            scale=30,
            bestEffort=True,
            maxPixels=1e9
        ).getInfo()

        return {
            "city": city,
            "year": year,
            "min_temp": round(stats.get('ST_B10_min'), 2),
            "max_temp": round(stats.get('ST_B10_max'), 2),
            "mean_temp": round(stats.get('ST_B10_mean'), 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 5. Wskaźnik UHI dla jednego roku: miasto vs pierścień referencyjny
@app.get("/api/uhi")
def get_uhi_index(city: str = "lublin", year: int = 2023):
    try:
        city_geom = get_city_geometry(city)
        rural_ring = get_reference_ring(city_geom)
        rural_mask = get_rural_mask()

        composite = get_landsat_composite(city_geom.union(rural_ring, ee.ErrorMargin(30)), year)
        if composite is None:
            raise HTTPException(status_code=404, detail=f"Brak danych dla roku {year}.")

        lst_celsius = compute_lst_celsius(composite)

        urban_temp = reduce_mean(lst_celsius, city_geom)
        rural_temp = reduce_mean(lst_celsius.updateMask(rural_mask), rural_ring)

        values = ee.Dictionary({'urban': urban_temp, 'rural': rural_temp}).getInfo()
        urban_val = values.get('urban')
        rural_val = values.get('rural')

        if urban_val is None or rural_val is None:
            raise HTTPException(status_code=404, detail="Brak wystarczających danych do wyliczenia wskaźnika UHI.")

        return {
            "city": city,
            "year": year,
            "urban_mean_temp": round(urban_val, 2),
            "rural_mean_temp": round(rural_val, 2),
            "uhi_index": round(urban_val - rural_val, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 6. Szereg czasowy wskaźnika UHI dla miasta
@app.get("/api/uhi/timeseries")
def get_uhi_timeseries(city: str = "lublin", start_year: int = 2014, end_year: int = 2023):
    try:
        city_geom = get_city_geometry(city)
        rural_ring = get_reference_ring(city_geom)
        rural_mask = get_rural_mask()
        combined_roi = city_geom.union(rural_ring, ee.ErrorMargin(30))

        series = []
        for year in range(start_year, end_year + 1):
            composite = get_landsat_composite(combined_roi, year)

            if composite is None:
                series.append({"year": year, "urban_mean_temp": None, "rural_mean_temp": None, "uhi_index": None})
                continue

            lst_celsius = compute_lst_celsius(composite)
            urban_temp = reduce_mean(lst_celsius, city_geom)
            rural_temp = reduce_mean(lst_celsius.updateMask(rural_mask), rural_ring)

            values = ee.Dictionary({'urban': urban_temp, 'rural': rural_temp}).getInfo()
            urban_val = values.get('urban')
            rural_val = values.get('rural')

            series.append({
                "year": year,
                "urban_mean_temp": round(urban_val, 2) if urban_val is not None else None,
                "rural_mean_temp": round(rural_val, 2) if rural_val is not None else None,
                "uhi_index": round(urban_val - rural_val, 2) if (urban_val is not None and rural_val is not None) else None
            })

        return {"city": city, "series": series}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 7. Granice miasta i obszaru referencyjnego jako GeoJSON (do narysowania na mapie)
@app.get("/api/geometry")
def get_city_and_reference_geometry(city: str = "lublin"):
    try:
        city_geom = get_city_geometry(city)
        rural_ring = get_reference_ring(city_geom)

        # Upraszczamy geometrię tylko na potrzeby wyświetlania - dużo mniejszy
        # i szybszy GeoJSON, bez wpływu na precyzję obliczeń w innych endpointach
        # (tam liczy się scale=30 w reduceRegion, nie dokładność samej geometrii).
        simplify_tolerance = ee.ErrorMargin(50)
        city_simplified = city_geom.simplify(simplify_tolerance)
        ring_simplified = rural_ring.simplify(simplify_tolerance)

        return {
            "city": city,
            "city_geojson": city_simplified.getInfo(),
            "reference_geojson": ring_simplified.getInfo()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 8. Pomocniczy endpoint diagnostyczny: lista nazw ADM2_NAME dla danego kraju w GAUL -
# przydatny do ustalenia poprawnej nazwy miasta w słowniku CITIES (np. dla Chełma).
@app.get("/api/debug/admin-names")
def list_admin_names(country: str = "Poland"):
    try:
        admin_boundaries = ee.FeatureCollection("FAO/GAUL/2015/level2")
        matches = admin_boundaries.filter(ee.Filter.eq('ADM0_NAME', country))
        names = matches.aggregate_array('ADM2_NAME').distinct().sort().getInfo()
        return {"country": country, "count": len(names), "adm2_names": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Oficjalna legenda i paleta ESA WorldCover v200 (pasmo 'Map').
WORLDCOVER_CLASSES = {
    10:  {"label": "Las",             "color": "006400"},
    20:  {"label": "Zarośla",         "color": "ffbb22"},
    30:  {"label": "Łąki",            "color": "ffff4c"},
    40:  {"label": "Pola uprawne",    "color": "f096ff"},
    50:  {"label": "Zabudowa",        "color": "fa0000"},
    60:  {"label": "Grunt odkryty",   "color": "b4b4b4"},
    70:  {"label": "Śnieg / lód",     "color": "f0f0f0"},
    80:  {"label": "Woda",            "color": "0064c8"},
    90:  {"label": "Mokradła",        "color": "0096a0"},
    95:  {"label": "Namorzyny",       "color": "00cf75"},
    100: {"label": "Mchy / porosty",  "color": "fae6a0"},
}


# 9. Szczegóły obszaru referencyjnego: pełne pokrycie terenu (WorldCover) w pierścieniu
# i rozkład temperatury w surowych pikselach 30m SAMEGO pierścienia (tylko piksele
# łąka/pole, które faktycznie wchodzą do liczenia UHI) - do zweryfikowania, czy obszar
# referencyjny jest faktycznie reprezentatywną okolicą wiejską, a nie czarną skrzynką.
@app.get("/api/reference")
def get_reference_details(city: str = "lublin", year: int = 2023):
    try:
        city_geom = get_city_geometry(city)
        rural_ring = get_reference_ring(city_geom)
        rural_mask = get_rural_mask()

        # --- Pokrycie terenu: PEŁNA klasyfikacja (wszystkie klasy), cały pierścień,
        # żeby było widać co dokładnie tam jest - nie tylko to, co przeszło maskę.
        worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
        landcover_clipped = worldcover.clip(rural_ring)

        # Kody klas WorldCover NIE są równo rozstawione (...80, 90, 95, 100), a getMapId
        # z paletą zakłada równe odstępy między min a max - bez tego remapu kolory
        # sąsiednich klas zlewałyby się (interpolacja), zamiast dać każdej klasie jej
        # własny, czysty kolor z legendy.
        wc_codes = list(WORLDCOVER_CLASSES.keys())
        landcover_indexed = landcover_clipped.remap(wc_codes, list(range(len(wc_codes))))
        lc_vis = {
            'min': 0, 'max': len(wc_codes) - 1,
            'palette': [WORLDCOVER_CLASSES[c]["color"] for c in wc_codes]
        }
        landcover_map_id = landcover_indexed.getMapId(lc_vis)

        histogram = landcover_clipped.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=rural_ring,
            scale=10,
            bestEffort=True,
            maxPixels=1e9
        ).getInfo()

        raw_hist = histogram.get('Map', {}) or {}
        total_px = sum(raw_hist.values()) or 1
        composition = []
        for code_str, count in sorted(raw_hist.items(), key=lambda kv: -kv[1]):
            code = int(code_str)
            info = WORLDCOVER_CLASSES.get(code, {"label": f"Klasa {code}", "color": "999999"})
            composition.append({
                "code": code,
                "label": info["label"],
                "color": info["color"],
                "percent": round(100 * count / total_px, 1),
                "used_as_reference": code in (30, 40)
            })

        # --- Temperatura: tylko piksele łąka/pole (ta sama maska co w /api/uhi), przycięte
        # do samego pierścienia - inna (węższa) skala kolorów niż na głównej mapie miasta,
        # bez wygładzania (surowe 30 m - to widok diagnostyczny, nie "ładny" podgląd).
        composite = get_landsat_composite(rural_ring, year)
        if composite is None:
            raise HTTPException(
                status_code=404,
                detail=f"Brak bezchmurnych danych dla roku {year} w obszarze referencyjnym."
            )

        lst_masked = compute_lst_celsius(composite).updateMask(rural_mask)
        lst_clipped = lst_masked.clip(rural_ring)

        lst_stats = lst_clipped.reduceRegion(
            reducer=ee.Reducer.mean()
                .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
                .combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
                .combine(reducer2=ee.Reducer.percentile([2, 98]), sharedInputs=True),
            geometry=rural_ring,
            scale=30,
            bestEffort=True,
            maxPixels=1e9
        ).getInfo()

        vis_min = lst_stats.get('ST_B10_p2')
        vis_max = lst_stats.get('ST_B10_p98')
        if vis_min is None or vis_max is None:
            raise HTTPException(
                status_code=404,
                detail="Zbyt mało danych temperaturowych w obszarze referencyjnym dla tego roku."
            )
        elif vis_max - vis_min < 2:
            vis_min, vis_max = vis_min - 1, vis_max + 1

        lst_vis_params = {
            'min': vis_min, 'max': vis_max,
            'palette': ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
        }
        lst_map_id = lst_clipped.getMapId(lst_vis_params)

        return {
            "city": city,
            "year": year,
            "landcover_url": landcover_map_id['tile_fetcher'].url_format,
            "landcover_composition": composition,
            "lst_url": lst_map_id['tile_fetcher'].url_format,
            "lst_vis_min": round(vis_min, 1),
            "lst_vis_max": round(vis_max, 1),
            "lst_stats": {
                "mean": round(lst_stats['ST_B10_mean'], 2) if lst_stats.get('ST_B10_mean') is not None else None,
                "median": round(lst_stats['ST_B10_median'], 2) if lst_stats.get('ST_B10_median') is not None else None,
                "min": round(lst_stats['ST_B10_min'], 2) if lst_stats.get('ST_B10_min') is not None else None,
                "max": round(lst_stats['ST_B10_max'], 2) if lst_stats.get('ST_B10_max') is not None else None,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
