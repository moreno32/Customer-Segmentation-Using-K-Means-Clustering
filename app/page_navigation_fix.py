"""
Page Navigation Fix

Este módulo proporciona funciones para resolver problemas de navegación 
entre páginas en la aplicación Streamlit de E-commerce Recommendation System.
"""

import streamlit as st
from time import sleep
import streamlit.components.v1 as components

def process_data_and_navigate(original_data, ref_date, clean_data_func, create_rfm_features_func, normalize_data_func):
    """
    Procesa los datos y navega directamente a la página de exploración.
    
    Parameters:
    -----------
    original_data : pd.DataFrame
        Datos originales a procesar
    ref_date : datetime
        Fecha de referencia para el análisis RFM
    clean_data_func : function
        Función para limpiar datos
    create_rfm_features_func : function
        Función para crear características RFM
    normalize_data_func : function
        Función para normalizar datos
    """
    # Directamente intenta procesar los datos
    try:
        # Un mensaje de procesamiento
        st.success("Procesando datos... Por favor espere.")
        
        # Procesar datos sin animaciones
        cleaned_df = clean_data_func(original_data)
        st.session_state.cleaned_data = cleaned_df
        
        rfm_df = create_rfm_features_func(cleaned_df, ref_date)
        st.session_state.rfm_data = rfm_df
        
        normalized_df, scaler = normalize_data_func(rfm_df)
        st.session_state.normalized_data = normalized_df
        st.session_state.scaler = scaler
        
        # Marcar como procesado
        st.session_state.data_processed = True
        
        # Marcar la siguiente página
        st.session_state.current_page = "Data Exploration"
        
        # Esto redirecciona directamente usando JavaScript
        js_code = """
        <script>
        // Establecer la página en los parámetros de URL
        const url = new URL(window.location.href);
        url.searchParams.set('page', 'Data Exploration');
        
        // Recargar con la nueva URL
        window.location.href = url;
        </script>
        """
        components.html(js_code, height=0)
        
        # Navegación alternativa si JavaScript falla
        st.text("Si no es redirigido automáticamente...")
        st.button("Haga clic aquí para ir a Exploración de Datos", on_click=lambda: st.rerun())
        
    except Exception as e:
        st.error(f"Error procesando datos: {str(e)}")
        
    return st.session_state.data_processed
