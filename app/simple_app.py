"""
Aplicación Streamlit simplificada con navegación básica
"""

import streamlit as st
import pandas as pd
import numpy as np

# Configuración de página
st.set_page_config(
    page_title="Navegación Simple",
    page_icon="🧭",
    layout="wide"
)

# Inicializar estado de sesión
if "page" not in st.session_state:
    st.session_state.page = "Página 1"

if "processed" not in st.session_state:
    st.session_state.processed = False

# Título principal
st.title("Aplicación de Navegación Simple")
st.write("---")

# Barra lateral con navegación
st.sidebar.title("Navegación")
pages = ["Página 1", "Página 2", "Página 3", "Página 4", "Página 5"]

# Radio button para navegación
selected_page = st.sidebar.radio("Ir a:", pages, index=pages.index(st.session_state.page))

# Actualizar página en el estado de sesión
if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

# Mostrar el contenido según la página seleccionada
if st.session_state.page == "Página 1":
    st.header("Página 1: Inicio")
    st.write("Esta es la página de inicio.")
    
    # Botón para procesar datos
    if st.button("Procesar Datos"):
        with st.spinner("Procesando..."):
            # Simular procesamiento
            import time
            time.sleep(2)
            st.session_state.processed = True
        st.success("¡Datos procesados correctamente!")
    
    # Mostrar botones de navegación
    st.write("---")
    st.write("### Navegación directa:")
    cols = st.columns(4)
    
    for i, page in enumerate(pages[1:], 1):
        with cols[i-1]:
            if st.button(f"Ir a {page}", key=f"btn_{page}"):
                st.session_state.page = page
                st.rerun()

elif st.session_state.page == "Página 2":
    st.header("Página 2: Datos")
    
    # Verificar si los datos han sido procesados
    if not st.session_state.processed:
        st.warning("¡Debes procesar los datos primero!")
        if st.button("Volver a la Página 1"):
            st.session_state.page = "Página 1"
            st.rerun()
    else:
        st.write("Aquí están los datos procesados.")
        
        # Generar datos de muestra
        data = pd.DataFrame({
            'x': np.random.rand(20),
            'y': np.random.rand(20)
        })
        
        # Mostrar datos
        st.dataframe(data)
        
        # Navegación
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Anterior"):
                st.session_state.page = "Página 1"
                st.rerun()
        with col2:
            if st.button("Siguiente →"):
                st.session_state.page = "Página 3"
                st.rerun()

elif st.session_state.page == "Página 3":
    st.header("Página 3: Visualización")
    
    # Verificar si los datos han sido procesados
    if not st.session_state.processed:
        st.warning("¡Debes procesar los datos primero!")
        if st.button("Volver a la Página 1"):
            st.session_state.page = "Página 1"
            st.rerun()
    else:
        # Crear gráfico de muestra
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['A', 'B', 'C'])
        
        st.line_chart(chart_data)
        
        # Navegación
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Anterior"):
                st.session_state.page = "Página 2"
                st.rerun()
        with col2:
            if st.button("Siguiente →"):
                st.session_state.page = "Página 4"
                st.rerun()

elif st.session_state.page == "Página 4":
    st.header("Página 4: Análisis")
    
    # Verificar si los datos han sido procesados
    if not st.session_state.processed:
        st.warning("¡Debes procesar los datos primero!")
        if st.button("Volver a la Página 1"):
            st.session_state.page = "Página 1"
            st.rerun()
    else:
        # Contenido de análisis
        st.write("Análisis de los datos procesados.")
        
        # Crear algunas métricas de ejemplo
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperatura", "70 °F", "1.2 °F")
        col2.metric("Viento", "9 mph", "-8%")
        col3.metric("Humedad", "86%", "4%")
        
        # Navegación
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Anterior"):
                st.session_state.page = "Página 3"
                st.rerun()
        with col2:
            if st.button("Siguiente →"):
                st.session_state.page = "Página 5"
                st.rerun()

elif st.session_state.page == "Página 5":
    st.header("Página 5: Conclusión")
    
    # Contenido final
    st.write("Has completado todos los pasos del proceso.")
    
    if st.button("Volver al inicio"):
        st.session_state.page = "Página 1"
        # Opcional: reiniciar otros estados
        st.session_state.processed = False
        st.rerun()
