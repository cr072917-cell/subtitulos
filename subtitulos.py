import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="Subtítulos para tus videos", page_icon="💬")

FUENTES = ["Montserrat", "Roboto", "Open Sans", "Liberation Sans", "DejaVu Sans"]
POSICIONES = ["Arriba", "Centro", "Abajo", "Personalizada"]
ANIMACIONES = ["Ninguna", "Pop", "Aparecer suave"]
IDIOMAS = {"Automático": None, "Español": "es", "Inglés": "en"}
ALTURA_POS = {"Arriba": 14, "Centro": 50, "Abajo": 84}

PRESETS = {
    "Reels": dict(palabras=3, mayus=True, fuente="Montserrat", negrita=True, tam=70, pos="Centro",
                  color="#FFFFFF", contorno=4, color_contorno="#000000", sombra=0, caja=False,
                  color_caja="#000000", opac_caja=60, resaltar=True, color_res="#FFD400", anim="Pop"),
    "Impacto": dict(palabras=2, mayus=True, fuente="Montserrat", negrita=True, tam=85, pos="Centro",
                    color="#FFFFFF", contorno=7, color_contorno="#000000", sombra=0, caja=False,
                    color_caja="#000000", opac_caja=60, resaltar=True, color_res="#FF3B3B", anim="Pop"),
    "Documental": dict(palabras=6, mayus=False, fuente="Open Sans", negrita=True, tam=48, pos="Abajo",
                       color="#FFFFFF", contorno=2, color_contorno="#000000", sombra=2, caja=False,
                       color_caja="#000000", opac_caja=60, resaltar=False, color_res="#FFD400",
                       anim="Aparecer suave"),
    "Minimal": dict(palabras=5, mayus=False, fuente="Roboto", negrita=False, tam=44, pos="Abajo",
                    color="#FFFFFF", contorno=0, color_contorno="#000000", sombra=3, caja=False,
                    color_caja="#000000", opac_caja=60, resaltar=False, color_res="#FFD400", anim="Ninguna"),
    "Clásico (con fondo)": dict(palabras=7, mayus=False, fuente="Liberation Sans", negrita=True, tam=50,
                                pos="Abajo", color="#FFFFFF", contorno=0, color_contorno="#000000",
                                sombra=0, caja=True, color_caja="#000000", opac_caja=65, resaltar=False,
                                color_res="#FFD400", anim="Ninguna"),
}

# Valores iniciales
if "preset" not in st.session_state:
    st.session_state["preset"] = "Reels"
for k, v in {**PRESETS["Reels"], "altura": 50}.items():
    st.session_state.setdefault(k, v)


def aplicar_preset():
    p = PRESETS.get(st.session_state["preset"])
    if p:
        st.session_state.update(p)


# ---------- Utilidades ----------

@st.cache_resource(show_spinner=False)
def cargar_modelo(nombre):
    return WhisperModel(nombre, device="cpu", compute_type="int8")


def dimensiones(ruta):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", ruta],
        capture_output=True, text=True, check=True,
    )
    s = json.loads(r.stdout)["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    rot = 0
    for sd in s.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if not rot:
        rot = int(s.get("tags", {}).get("rotate", 0) or 0)
    if abs(rot) % 180 == 90:
        w, h = h, w
    return w, h


def audio_a_array(ruta):
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", ruta, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def transcribir(ruta, calidad, idioma):
    audio = audio_a_array(ruta)
    if audio.size == 0:
        return []
    modelo = cargar_modelo(calidad)
    segmentos, _ = modelo.transcribe(audio, language=idioma, word_timestamps=True, vad_filter=True)
    palabras = []
    for seg in segmentos:
        for w in seg.words or []:
            t = w.word.strip()
            if t:
                palabras.append((t, w.start, w.end))
    return palabras


def agrupar(palabras, n):
    grupos, actual = [], []
    for p in palabras:
        if actual and (len(actual) >= n or p[1] - actual[-1][2] > 0.8):
            grupos.append(actual)
            actual = []
        actual.append(p)
    if actual:
        grupos.append(actual)
    return grupos


def limpiar(t, mayus):
    t = t.replace("{", "").replace("}", "").replace("\\", "")
    return t.upper() if mayus else t


def t_ass(s):
    cs = int(round(max(s, 0) * 100))
    h, resto = divmod(cs, 360000)
    m, resto = divmod(resto, 6000)
    sec, c = divmod(resto, 100)
    return f"{h}:{m:02d}:{sec:02d}.{c:02d}"


def t_srt(s):
    ms = int(round(max(s, 0) * 1000))
    h, resto = divmod(ms, 3600000)
    m, resto = divmod(resto, 60000)
    sec, ms = divmod(resto, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def ass_style_color(hex_color, alpha=0):
    r, g, b = rgb(hex_color)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def ass_inline_color(hex_color):
    r, g, b = rgb(hex_color)
    return f"&H{b:02X}{g:02X}{r:02X}&"


def construir_ass(grupos, o, ancho, alto):
    pw = 1080
    ph = round(1080 * alto / ancho)
    pct = o["altura"] if o["pos"] == "Personalizada" else ALTURA_POS[o["pos"]]
    x, y = pw // 2, int(ph * pct / 100)

    caja = o["caja"]
    border_style = 3 if caja else 1
    grosor = 12 if caja else o["contorno"]
    sombra = 0 if caja else o["sombra"]
    if caja:
        alpha = round(255 * (1 - o["opac_caja"] / 100))
        col_contorno = ass_style_color(o["color_caja"], alpha)
    else:
        col_contorno = ass_style_color(o["color_contorno"])

    estilo = (
        f"Style: Default,{o['fuente']},{o['tam']},{ass_style_color(o['color'])},"
        f"{ass_style_color(o['color'])},{col_contorno},&H80000000,"
        f"{-1 if o['negrita'] else 0},0,0,0,100,100,0,0,{border_style},{grosor},{sombra},5,60,60,60,1"
    )

    anim_txt = {
        "Pop": r"\fscx80\fscy80\t(0,150,\fscx100\fscy100)",
        "Aparecer suave": r"\fad(150,0)",
        "Ninguna": "",
    }[o["anim"]]

    cabecera = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {pw}\nPlayResY: {ph}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{estilo}\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    def dialogo(a, b, anim, texto):
        return f"Dialogue: 0,{t_ass(a)},{t_ass(b)},Default,,0,0,0,,{{\\an5\\pos({x},{y}){anim}}}{texto}"

    prim = ass_inline_color(o["color"])
    res = ass_inline_color(o["color_res"])
    lineas = []
    for gi, g in enumerate(grupos):
        ini = g[0][1]
        fin = g[-1][2] + 0.15
        if gi + 1 < len(grupos):
            fin = min(fin, grupos[gi + 1][0][1])
        txts = [limpiar(p[0], o["mayus"]) for p in g]
        if o["resaltar"]:
            for i in range(len(g)):
                a = ini if i == 0 else g[i][1]
                b = g[i + 1][1] if i + 1 < len(g) else fin
                partes = []
                for j, t in enumerate(txts):
                    if j == i:
                        partes.append(f"{{\\1c{res}}}{t}{{\\1c{prim}}}")
                    else:
                        partes.append(t)
                lineas.append(dialogo(a, b, anim_txt if i == 0 else "", " ".join(partes)))
        else:
            lineas.append(dialogo(ini, fin, anim_txt, " ".join(txts)))
    return cabecera + "\n".join(lineas) + "\n"


def construir_srt(grupos, mayus):
    bloques = []
    for i, g in enumerate(grupos, 1):
        texto = " ".join(limpiar(p[0], mayus) for p in g)
        bloques.append(f"{i}\n{t_srt(g[0][1])} --> {t_srt(g[-1][2])}\n{texto}\n")
    return "\n".join(bloques)


# ---------- Interfaz ----------

st.title("💬 Subtítulos para tus videos")
st.write("Sube tu video, la app escucha el audio y le pega los subtítulos.")

archivo = st.file_uploader("Sube tu video", type=["mp4", "mov", "mkv", "webm"])

st.selectbox("Estilo rápido", ["Personalizado"] + list(PRESETS), key="preset", on_change=aplicar_preset)

calidad = st.selectbox(
    "Calidad de escucha", ["tiny", "base", "small", "medium"], index=1,
    help="Más grande = más preciso pero más lento. 'base' es un buen equilibrio.",
)
idioma = st.selectbox("Idioma del audio", list(IDIOMAS))

st.subheader("Texto")
st.slider("Palabras por subtítulo", 1, 10, key="palabras")
st.checkbox("TODO EN MAYÚSCULAS", key="mayus")
st.selectbox("Fuente", FUENTES, key="fuente")
st.checkbox("Negrita", key="negrita")
st.slider("Tamaño de letra", 20, 150, key="tam")
st.color_picker("Color del texto", key="color")

st.subheader("Posición")
st.selectbox("Posición", POSICIONES, key="pos")
if st.session_state["pos"] == "Personalizada":
    st.slider("Altura (0% arriba, 100% abajo)", 5, 95, key="altura")

st.subheader("Contorno, sombra y fondo")
st.slider("Grosor del contorno", 0, 12, key="contorno")
st.color_picker("Color del contorno", key="color_contorno")
st.slider("Sombra", 0, 10, key="sombra")
st.checkbox("Fondo detrás del texto (ignora el contorno)", key="caja")
if st.session_state["caja"]:
    st.color_picker("Color del fondo", key="color_caja")
    st.slider("Opacidad del fondo (%)", 10, 100, key="opac_caja")

st.subheader("Efectos")
st.checkbox("Resaltar la palabra que se dice", key="resaltar")
if st.session_state["resaltar"]:
    st.color_picker("Color de la palabra resaltada", key="color_res")
st.selectbox("Animación", ANIMACIONES, key="anim")

CLAVES = ["palabras", "mayus", "fuente", "negrita", "tam", "pos", "altura", "color", "contorno",
          "color_contorno", "sombra", "caja", "color_caja", "opac_caja", "resaltar", "color_res", "anim"]

if st.button("Crear subtítulos", type="primary", disabled=archivo is None):
    opciones = {k: st.session_state[k] for k in CLAVES}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ext = Path(archivo.name).suffix or ".mp4"
            entrada = tmp / f"entrada{ext}"
            entrada.write_bytes(archivo.getbuffer())

            with st.spinner("Escuchando el audio..."):
                palabras = transcribir(str(entrada), calidad, IDIOMAS[idioma])
            if not palabras:
                st.warning("No encontré voz en el video.")
                st.stop()

            grupos = agrupar(palabras, opciones["palabras"])
            ancho, alto = dimensiones(str(entrada))
            (tmp / "subs.ass").write_text(construir_ass(grupos, opciones, ancho, alto), encoding="utf-8")

            with st.spinner("Pegando los subtítulos al video..."):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", entrada.name, "-vf", "ass=subs.ass",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "salida.mp4"],
                    cwd=tmp, check=True, capture_output=True,
                )
            st.session_state["resultado"] = {
                "video": (tmp / "salida.mp4").read_bytes(),
                "srt": construir_srt(grupos, opciones["mayus"]),
            }
    except subprocess.CalledProcessError as e:
        st.error("Algo falló al procesar el video.")
        st.code((e.stderr or b"").decode(errors="ignore")[-1500:])

if "resultado" in st.session_state:
    st.success("¡Listo!")
    st.video(st.session_state["resultado"]["video"])
    st.download_button("⬇️ Descargar video", st.session_state["resultado"]["video"],
                       file_name="video_con_subtitulos.mp4", mime="video/mp4")
    st.download_button("⬇️ Descargar subtítulos (.srt)", st.session_state["resultado"]["srt"],
                       file_name="subtitulos.srt", mime="text/plain")
