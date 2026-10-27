import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import unicodedata
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Observatorio Departamental - Jhon Ortiz",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# NORMALIZACIÓN
# ============================================================
def quitar_tildes(texto):
    if pd.isna(texto) or texto is None:
        return ""
    t = str(texto).strip().lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return t.replace(",", "").replace(".", " ").replace("  ", " ").strip()

DEPARTAMENTOS_CANONICOS = {
    "amazonas": "Amazonas", "antioquia": "Antioquia", "arauca": "Arauca",
    "atlantico": "Atlántico", "bogota": "Bogotá D.C.", "bogota d c": "Bogotá D.C.",
    "bogota dc": "Bogotá D.C.", "bolivar": "Bolívar", "boyaca": "Boyacá",
    "caldas": "Caldas", "caqueta": "Caquetá", "casanare": "Casanare",
    "cauca": "Cauca", "cesar": "Cesar", "choco": "Chocó", "cordoba": "Córdoba",
    "cundinamarca": "Cundinamarca", "guainia": "Guainía", "guaviare": "Guaviare",
    "huila": "Huila", "la guajira": "La Guajira", "magdalena": "Magdalena",
    "meta": "Meta", "narino": "Nariño", "norte de santander": "Norte de Santander",
    "putumayo": "Putumayo", "quindio": "Quindío", "risaralda": "Risaralda",
    "san andres": "San Andrés y Providencia", "san andres y providencia": "San Andrés y Providencia",
    "santander": "Santander", "sucre": "Sucre", "tolima": "Tolima",
    "valle": "Valle del Cauca", "valle del cauca": "Valle del Cauca",
    "vaupes": "Vaupés", "vichada": "Vichada"
}

def normalizar_departamento(nombre):
    clave = quitar_tildes(nombre)
    return DEPARTAMENTOS_CANONICOS.get(clave, str(nombre).strip().title() if nombre else None)

# ============================================================
# CARGA DE DATOS (con cache)
# ============================================================
@st.cache_data(ttl=3600, show_spinner="Cargando datos oficiales...")
def cargar_datos():
    def cargar_pib_sectores():
        try:
            url = "https://www.datos.gov.co/resource/kgyi-qc7j.json?$limit=50000&$order=a_o DESC"
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            df = pd.DataFrame(r.json())
            df["anio"] = pd.to_numeric(df["a_o"], errors="coerce")
            df["valor"] = pd.to_numeric(df["valor_miles_de_millones_de"], errors="coerce")
            df["departamento"] = df["departamento"].apply(normalizar_departamento)
            df = df[df["tipo_de_precios"].str.contains("constantes", case=False, na=False)]
            
            resultados = []
            total = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum()
            total["indicador"] = "PIB Total (miles de millones constantes 2015)"
            total["fuente"] = "DANE"
            resultados.append(total)
            
            if "sector" in df.columns:
                for sector, nombre in [
                    ("Primario", "PIB Sector Primario"),
                    ("Secundario", "PIB Sector Secundario"),
                    ("Terciario", "PIB Sector Terciario")
                ]:
                    temp = df[df["sector"].str.contains(sector, case=False, na=False)]
                    if not temp.empty:
                        agg = temp.groupby(["departamento", "anio"], as_index=False)["valor"].sum()
                        agg["indicador"] = f"{nombre} (miles de millones constantes 2015)"
                        agg["fuente"] = "DANE"
                        resultados.append(agg)
            return pd.concat(resultados, ignore_index=True).dropna()
        except:
            return pd.DataFrame()

    def cargar_dnp():
        try:
            url = "https://www.datos.gov.co/resource/wc86-9j4a.json?$limit=80000"
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            df = pd.DataFrame(r.json())
            df["departamento"] = df["departamento"].apply(normalizar_departamento)
            df["anio"] = pd.to_numeric(df.get("vigencia"), errors="coerce")
            valor_col = next((c for c in ["valorpagado", "valorobligado", "valorcomprometido"] if c in df.columns), None)
            if not valor_col:
                return pd.DataFrame()
            df["valor"] = pd.to_numeric(df[valor_col], errors="coerce")
            agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum()
            agg["indicador"] = "Inversión pública DNP (COP)"
            agg["fuente"] = "DNP"
            return agg.dropna()
        except:
            return pd.DataFrame()

    def cargar_educacion():
        try:
            url = "https://www.datos.gov.co/resource/ji8i-4anb.json?$limit=10000"
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            df = pd.DataFrame(r.json())
            anio_col = next((c for c in ["ano", "a_o", "año"] if c in df.columns), None)
            if anio_col is None:
                return pd.DataFrame()
            df["anio"] = pd.to_numeric(df[anio_col], errors="coerce")
            df["departamento"] = df["departamento"].apply(normalizar_departamento)
            resultados = []
            for col, nombre in [
                ("cobertura_neta", "Cobertura neta educación (%)"),
                ("cobertura_bruta", "Cobertura bruta educación (%)"),
                ("tasa_matriculacion_5_16", "Tasa de matrícula 5-16 años (%)")
            ]:
                if col in df.columns:
                    temp = df.groupby(["departamento", "anio"], as_index=False)[col].mean().dropna()
                    temp = temp.rename(columns={col: "valor"})
                    temp["indicador"] = nombre
                    temp["fuente"] = "MEN"
                    resultados.append(temp)
            return pd.concat(resultados, ignore_index=True) if resultados else pd.DataFrame()
        except:
            return pd.DataFrame()

    partes = []
    for fn in [cargar_pib_sectores, cargar_dnp, cargar_educacion]:
        d = fn()
        if not d.empty:
            partes.append(d)
    
    if not partes:
        return pd.DataFrame(columns=["departamento", "anio", "valor", "indicador", "fuente"])
    
    maestro = pd.concat(partes, ignore_index=True)
    maestro["anio"] = maestro["anio"].astype(int)
    return maestro.dropna(subset=["valor", "departamento", "anio"]).sort_values(["indicador", "departamento", "anio"]).reset_index(drop=True)

@st.cache_data
def cargar_geojson():
    try:
        url = "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except:
        return None

# ============================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================
def filtrar(df, deptos, indicador, anio_min, anio_max):
    d = df.copy()
    if deptos and "Todos" not in deptos:
        d = d[d["departamento"].isin(deptos)]
    if indicador and indicador != "Todos":
        d = d[d["indicador"] == indicador]
    d = d[(d["anio"] >= anio_min) & (d["anio"] <= anio_max)]
    return d

def mapa_coropletico(df, indicador, anio, geojson):
    d = df[(df["indicador"] == indicador) & (df["anio"] == anio)].copy()
    if d.empty or geojson is None:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos o GeoJSON no disponible", x=0.5, y=0.5, showarrow=False)
        return fig

    mapeo = {
        "Amazonas": "AMAZONAS", "Antioquia": "ANTIOQUIA", "Arauca": "ARAUCA",
        "Atlántico": "ATLÁNTICO", "Bogotá D.C.": "BOGOTÁ, D.C.", "Bolívar": "BOLÍVAR",
        "Boyacá": "BOYACÁ", "Caldas": "CALDAS", "Caquetá": "CAQUETÁ",
        "Casanare": "CASANARE", "Cauca": "CAUCA", "Cesar": "CESAR",
        "Chocó": "CHOCÓ", "Córdoba": "CÓRDOBA", "Cundinamarca": "CUNDINAMARCA",
        "Guainía": "GUAINÍA", "Guaviare": "GUAVIARE", "Huila": "HUILA",
        "La Guajira": "LA GUAJIRA", "Magdalena": "MAGDALENA", "Meta": "META",
        "Nariño": "NARIÑO", "Norte de Santander": "NORTE DE SANTANDER",
        "Putumayo": "PUTUMAYO", "Quindío": "QUINDÍO", "Risaralda": "RISARALDA",
        "San Andrés y Providencia": "ARCHIPIÉLAGO DE SAN ANDRÉS",
        "Santander": "SANTANDER", "Sucre": "SUCRE", "Tolima": "TOLIMA",
        "Valle del Cauca": "VALLE DEL CAUCA", "Vaupés": "VAUPÉS", "Vichada": "VICHADA"
    }
    
    d["location"] = d["departamento"].map(mapeo)
    d = d.dropna(subset=["location"])
    
    fig = px.choropleth(
        d, geojson=geojson, locations="location",
        featureidkey="properties.DPTO_CNMBR", color="valor",
        hover_name="departamento", color_continuous_scale="Blues",
        title=f"Mapa Coroplético – {indicador} ({anio})"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
    return fig

# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
st.title("🇨🇴 Observatorio de Actividad Económica y Social por Departamentos")
st.markdown("**Creado por: Jhon Ortiz** | Seminario de Investigación | Fuentes: DANE · DNP · MEN")

with st.spinner("Cargando datos..."):
    DF = cargar_datos()
    GEOJSON = cargar_geojson()

if DF.empty:
    st.error("No se pudieron cargar los datos. Intenta recargar la página.")
    st.stop()

lista_deptos = sorted(DF["departamento"].dropna().unique())
lista_inds = sorted(DF["indicador"].unique())
amin = int(DF["anio"].min())
amax = min(int(DF["anio"].max()), 2026)

# Sidebar
st.sidebar.header("Filtros")
deptos = st.sidebar.multiselect("Departamento(s)", ["Todos"] + lista_deptos, default=["Antioquia", "Bogotá D.C.", "Valle del Cauca"])
indicador = st.sidebar.selectbox("Indicador", ["Todos"] + lista_inds)
anio_min = st.sidebar.slider("Año inicial", 2000, 2026, max(amin, 2012))
anio_max = st.sidebar.slider("Año final", 2000, 2026, amax)
anio_mapa = st.sidebar.slider("Año para Mapa y Barras", 2000, 2026, amax)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Tendencias", "2. Mapa Coroplético", "3. EDA", 
    "4. Barras", "5. Correlación", "6. PCA", "7. Datos"
])

ind_real = indicador if indicador != "Todos" else lista_inds[0]

with tab1:
    d = filtrar(DF, deptos, indicador, anio_min, anio_max)
    if not d.empty:
        fig = px.line(d, x="anio", y="valor", color="departamento", markers=True,
                      title=f"Tendencia – {indicador}", template="plotly_white")
        fig.update_layout(hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Sin datos para los filtros seleccionados.")

with tab2:
    fig = mapa_coropletico(DF, ind_real, anio_mapa, GEOJSON)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    d = filtrar(DF, deptos, indicador, anio_min, anio_max)
    if not d.empty:
        st.write(f"**Observaciones:** {len(d):,} | **Departamentos:** {d['departamento'].nunique()}")
        st.dataframe(d.groupby("indicador")["valor"].describe().round(2))
        fig = px.box(d, x="indicador", y="valor", color="indicador", template="plotly_white")
        fig.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Sin datos.")

with tab4:
    d = DF[(DF["indicador"] == ind_real) & (DF["anio"] == anio_mapa)]
    if not d.empty:
        d = d.nlargest(15, "valor")
        fig = px.bar(d, x="valor", y="departamento", orientation="h",
                     title=f"Top 15 – {ind_real} ({anio_mapa})",
                     color="valor", color_continuous_scale="Blues", template="plotly_white")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Sin datos.")

with tab5:
    inds = DF["indicador"].unique().tolist()
    panel = DF[DF["indicador"].isin(inds) & (DF["anio"] >= anio_min) & (DF["anio"] <= anio_max)]
    panel = panel.pivot_table(index=["departamento", "anio"], columns="indicador", values="valor").reset_index()
    if len(panel) > 30:
        num_cols = [c for c in panel.columns if c not in ["departamento", "anio"]]
        corr = panel[num_cols].corr()
        fig1 = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Matriz de Correlación")
        st.plotly_chart(fig1, use_container_width=True)
        
        X = StandardScaler().fit_transform(panel[num_cols].fillna(panel[num_cols].median()))
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        panel["Cluster"] = kmeans.fit_predict(X)
        fig2 = px.scatter(panel, x=num_cols[0], y=num_cols[1], color="Cluster",
                          hover_data=["departamento", "anio"], title="Clustering (K-Means)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Datos insuficientes para correlación.")

with tab6:
    inds = DF["indicador"].unique().tolist()
    panel = DF[DF["indicador"].isin(inds) & (DF["anio"] >= anio_min) & (DF["anio"] <= anio_max)]
    panel = panel.pivot_table(index=["departamento", "anio"], columns="indicador", values="valor").reset_index()
    if len(panel) > 20:
        num_cols = [c for c in panel.columns if c not in ["departamento", "anio"]]
        X = StandardScaler().fit_transform(panel[num_cols].fillna(0))
        pca = PCA(n_components=2)
        comps = pca.fit_transform(X)
        panel["PC1"] = comps[:, 0]
        panel["PC2"] = comps[:, 1]
        fig = px.scatter(panel, x="PC1", y="PC2", color="departamento", hover_data=["anio"],
                         title=f"PCA – Varianza explicada: {pca.explained_variance_ratio_.sum():.1%}")
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Varianza explicada por PC1 + PC2: **{pca.explained_variance_ratio_.sum():.1%}**")
    else:
        st.warning("Datos insuficientes para PCA.")

with tab7:
    st.markdown("""
    ### Fuentes oficiales
    | Indicador | Fuente |
    |-----------|--------|
    | PIB Total, Primario, Secundario y Terciario | DANE |
    | Inversión pública | DNP |
    | Cobertura educación + Tasa de matrícula | MEN |
    """)
    st.dataframe(filtrar(DF, deptos, indicador, anio_min, anio_max).head(400))

st.markdown("---")
st.caption("Observatorio Departamental de Colombia · Desarrollado por **Jhon Ortiz** · 2025")
