import os
import shutil
import subprocess
import tempfile

import numpy as np
import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="Subtítulos para tus videos", page_icon="💬")
st.title("💬 Subtítulos para tus videos")
st.caption("Sube tu video, la app escucha el audio y le pega los subtítulos.")


def ruta_ffmpeg():
    ruta = shutil.which("ffmpeg")
    if ruta:
        return ruta
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


@st.cache_resource
def cargar_modelo(nombre):
    return WhisperModel(nombre, device="cpu", compute_type="int8")


def cargar_audio(ruta_video, ff):
    r = subprocess.run(
        [ff, "-nostdin", "-i", ruta_video, "-vn", "-ac", "1", "-ar", "16000",
         "-f", "f32le", "-"],
        capture_output=True,
    )
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("No pude sacar el audio del video.")
    return np.frombuffer(r.stdout, dtype=np.float32)


def tiempo_srt(seg):
    ms = int(round(seg * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def crear_srt(palabras, por_grupo, mayusculas):
    bloques = []
    for i in range(0, len(palabras), por_grupo):
        grupo = palabras[i : i + por_grupo]
        texto = " ".join(p.word.strip() for p in grupo)
        if mayusculas:
            texto = texto.upper()
        inicio = grupo[0].start
        fin = grupo[-1].end
        if i + por_grupo < len(palabras):
            siguiente = palabras[i + por_grupo].start
            if 0 <= siguiente - fin < 0.4:
                fin = siguiente
        bloques.append(
            f"{len(bloques) + 1}\n{tiempo_srt(inicio)} --> {tiempo_srt(fin)}\n{texto}\n"
        )
    return "\n".join(bloques)


video = st.file_uploader("Sube tu video", type=["mp4", "mov", "mkv", "webm"])

col1, col2 = st.columns(2)
with col1:
    modelo = st.selectbox(
        "Calidad de escucha",
        ["tiny", "base", "small"],
        index=1,
        help="tiny es más rápido, small entiende mejor pero tarda más.",
    )
    por_grupo = st.slider("Palabras por subtítulo", 1, 6, 3)
    mayusculas = st.checkbox("TODO EN MAYÚSCULAS", value=True)
with col2:
    tamano = st.slider("Tamaño de letra", 8, 30, 16)
    posicion = st.selectbox("Posición", ["Centro", "Abajo", "Arriba"])
    color = st.selectbox("Color", ["Blanco", "Amarillo"])

# Paso 1: escuchar el audio y crear los subtítulos
if st.button("1. Crear subtítulos", type="primary", disabled=video is None):
    ff = ruta_ffmpeg()
    if not ff:
        st.error("No encontré ffmpeg.")
        st.stop()
    with tempfile.TemporaryDirectory() as tmp:
        ext = os.path.splitext(video.name)[1] or ".mp4"
        entrada = os.path.join(tmp, "entrada" + ext)
        with open(entrada, "wb") as f:
            f.write(video.getbuffer())
        with st.spinner("Escuchando el audio (la primera vez descarga el modelo)..."):
            try:
                audio = cargar_audio(entrada, ff)
                segmentos, _ = cargar_modelo(modelo).transcribe(
                    audio, language="es", word_timestamps=True, vad_filter=True
                )
                palabras = [w for s in segmentos for w in (s.words or [])]
            except Exception as e:
                st.error(f"No se pudo escuchar el audio: {e}")
                st.stop()
    if not palabras:
        st.warning("No se escuchó ninguna voz en el video.")
    else:
        st.session_state["srt"] = crear_srt(palabras, por_grupo, mayusculas)
        st.session_state.pop("video_final", None)

# Paso 2: corregir y pegar al video
if "srt" in st.session_state and video is not None:
    st.subheader("Revisa los subtítulos")
    st.caption("Corrige solo las palabras mal escritas. No cambies los números ni los tiempos.")
    srt_editado = st.text_area("Subtítulos", st.session_state["srt"], height=300)

    if st.button("2. Pegar subtítulos al video", type="primary"):
        ff = ruta_ffmpeg()
        if not ff:
            st.error("No encontré ffmpeg.")
            st.stop()

        alineacion = {"Abajo": 2, "Centro": 5, "Arriba": 8}[posicion]
        color_hex = {"Blanco": "&H00FFFFFF", "Amarillo": "&H0000FFFF"}[color]
        estilo = (
            f"FontName=Arial,FontSize={tamano},Bold=1,PrimaryColour={color_hex},"
            f"OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0,"
            f"Alignment={alineacion},MarginV=60"
        )

        with tempfile.TemporaryDirectory() as tmp:
            ext = os.path.splitext(video.name)[1] or ".mp4"
            with open(os.path.join(tmp, "entrada" + ext), "wb") as f:
                f.write(video.getbuffer())
            with open(os.path.join(tmp, "subs.srt"), "w", encoding="utf-8") as f:
                f.write(srt_editado)

            with st.spinner("Pegando los subtítulos al video..."):
                r = subprocess.run(
                    [
                        ff, "-y", "-i", "entrada" + ext,
                        "-vf", f"subtitles=subs.srt:force_style='{estilo}'",
                        "-c:v", "libx264", "-preset", "veryfast",
                        "-c:a", "aac", "salida.mp4",
                    ],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                )
            if r.returncode != 0:
                st.error("Falló al pegar los subtítulos.")
                st.code(r.stderr[-1500:])
            else:
                with open(os.path.join(tmp, "salida.mp4"), "rb") as f:
                    st.session_state["video_final"] = f.read()

if "video_final" in st.session_state:
    st.success("¡Listo!")
    st.video(st.session_state["video_final"])
    st.download_button(
        "⬇️ Descargar video con subtítulos",
        data=st.session_state["video_final"],
        file_name="video_con_subtitulos.mp4",
        mime="video/mp4",
    )
