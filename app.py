# app.py
# Caixa - Fechamento Online (Streamlit)
# ------------------------------------------------------------
# Rodar local:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Deploy (Streamlit Community Cloud):
#   - Suba estes arquivos em um repo GitHub
#   - Aponte o deploy para app.py
# ------------------------------------------------------------

import sqlite3
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "fechamentos.db"
OPERADORES_CSV = APP_DIR / "operadores.csv"

st.set_page_config(
    page_title="Fechamento do Caixa",
    layout="wide",
)

# ==============================
# Utilitários (mesma lógica do Tkinter, adaptada)
# ==============================

def formatar_brasileiro(valor: float) -> str:
    try:
        valor = float(valor)
    except Exception:
        valor = 0.0
    s = f"{valor:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def br_to_float(texto) -> float:
    if texto is None:
        return 0.0

    # Trata NaN
    try:
        import math
        if isinstance(texto, float) and math.isnan(texto):
            return 0.0
    except Exception:
        pass

    if isinstance(texto, (int, float)):
        try:
            return float(texto)
        except Exception:
            return 0.0

    t = str(texto).strip()
    if not t:
        return 0.0

    t = t.replace("R$", "").replace(" ", "").strip()

    if "," in t and "." in t:
        # Decide o separador decimal pelo último que aparece
        if t.rfind(",") > t.rfind("."):
            # 1.234,56 -> 1234.56
            t = t.replace(".", "").replace(",", ".")
        else:
            # 1,234.56 -> 1234.56
            t = t.replace(",", "")
    elif "," in t:
        # 1234,56 -> 1234.56  (remove possíveis pontos de milhar)
        t = t.replace(".", "").replace(",", ".")
    else:
        # 1234.56 -> 1234.56 (mantém)
        pass

    try:
        return float(t)
    except ValueError:
        return 0.0


def validar_operadores_df(df: pd.DataFrame) -> tuple[bool, str, list[str]]:
    # Espera header "nome" como no seu arquivo original
    if df is None or df.empty:
        return False, "Arquivo vazio.", []
    cols = [c.strip().lower() for c in df.columns.tolist()]
    if cols != ["nome"]:
        return False, f"Cabeçalho inválido. Esperado ['nome'], encontrado {df.columns.tolist()}", []
    nomes = [str(x).strip() for x in df["nome"].tolist()]
    if any(n == "" for n in nomes):
        return False, "Operador com nome vazio encontrado.", []
    if len(nomes) != len(set(nomes)):
        return False, "Nomes de operadores duplicados encontrados.", []
    return True, "OK", nomes

# ==============================
# Banco de dados
# ==============================

def db_connect():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fechamentos (
            data TEXT NOT NULL,
            nome_operador TEXT NOT NULL,
            moedas_fisico REAL,
            dinheiro_fisico REAL,
            cartao_fisico REAL,
            pix_fisico REAL,
            entrega_fisico REAL,
            sangria_fisico REAL,
            sistema_dinheiro REAL,
            sistema_cartao REAL,
            sistema_pix REAL,
            sistema_entrega REAL,
            sistema_troco REAL,
            dif_dinheiro REAL,
            dif_cartao REAL,
            dif_pix REAL,
            dif_entrega REAL,
            total_geral REAL,
            PRIMARY KEY (data, nome_operador)
        )
    """)
    conn.commit()
    return conn

@st.cache_resource
def get_conn():
    return db_connect()

def salvar_fechamento(payload: dict):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO fechamentos VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        payload["data"],
        payload["nome_operador"],
        payload["moedas_fisico"],
        payload["dinheiro_fisico"],
        payload["cartao_fisico"],
        payload["pix_fisico"],
        payload["entrega_fisico"],
        payload["sangria_fisico"],
        payload["sistema_dinheiro"],
        payload["sistema_cartao"],
        payload["sistema_pix"],
        payload["sistema_entrega"],
        payload["sistema_troco"],
        payload["dif_dinheiro"],
        payload["dif_cartao"],
        payload["dif_pix"],
        payload["dif_entrega"],
        payload["total_geral"],
    ))
    conn.commit()

def listar_fechamentos(yyyy_mm_dd: str | None = None) -> pd.DataFrame:
    conn = get_conn()
    if yyyy_mm_dd:
        df = pd.read_sql_query(
            "SELECT * FROM fechamentos WHERE data = ? ORDER BY nome_operador",
            conn,
            params=(yyyy_mm_dd,),
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM fechamentos ORDER BY data DESC, nome_operador",
            conn,
        )
    return df

# ==============================
# Operadores (CSV no diretório do app)
# ==============================

def carregar_operadores() -> list[str]:
    if not OPERADORES_CSV.exists():
        return []
    try:
        df = pd.read_csv(OPERADORES_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        # tenta latin-1 para reduzir atrito (mas mantém validação de coluna)
        df = pd.read_csv(OPERADORES_CSV, encoding="latin-1")
    ok, msg, nomes = validar_operadores_df(df)
    if not ok:
        st.sidebar.error(f"operadores.csv inválido: {msg}")
        return []
    return nomes

def salvar_operadores_csv(nomes: list[str]):
    with open(OPERADORES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nome"])
        for n in nomes:
            w.writerow([n])

# ==============================
# Estado
# ==============================

if "operadores" not in st.session_state:
    st.session_state.operadores = carregar_operadores()

# Data do dia (mesma ideia do Tkinter)
data_dia_br = datetime.now().strftime("%d/%m/%Y")
data_sqlite = datetime.now().strftime("%Y-%m-%d")

st.title(f"Fechamento do Caixa — {data_dia_br}")

# Sidebar: operadores + upload
with st.sidebar:
    st.header("Configuração")
    st.caption("Operadores via operadores.csv (UTF-8, coluna única: nome).")

    if st.session_state.operadores:
        st.success(f"{len(st.session_state.operadores)} operador(es) carregado(s).")
    else:
        st.warning("Nenhum operador carregado ainda.")

    up = st.file_uploader("Enviar operadores.csv", type=["csv"])
    if up is not None:
        try:
            df_up = pd.read_csv(up, encoding="utf-8")
        except UnicodeDecodeError:
            df_up = pd.read_csv(up, encoding="latin-1")
        ok, msg, nomes = validar_operadores_df(df_up)
        if ok:
            st.session_state.operadores = nomes
            salvar_operadores_csv(nomes)
            st.success("Operadores atualizados.")
        else:
            st.error(msg)

    st.divider()
    st.subheader("Histórico (admin)")
    filtro_data = st.text_input("Filtrar por data (YYYY-MM-DD)", value=data_sqlite)
    if st.button("Carregar fechamentos"):
        st.session_state._hist = listar_fechamentos(filtro_data.strip() or None)

# ==============================
# Layout principal
# ==============================

col_moedas, col_dinheiro, col_cartao, col_pix = st.columns([1.1, 1.1, 0.8, 1.2])
col_sangria, col_entrega, col_sistema = st.columns([1.2, 1.2, 1.6])

# ---- Moedas
with col_moedas:
    st.subheader("🪙 Moedas (físico)")
    qtd_1   = st.number_input("Qtd 1,00", min_value=0, step=1, value=0, key="qtd_m_1")
    qtd_50  = st.number_input("Qtd 0,50", min_value=0, step=1, value=0, key="qtd_m_50")
    qtd_25  = st.number_input("Qtd 0,25", min_value=0, step=1, value=0, key="qtd_m_25")
    qtd_10  = st.number_input("Qtd 0,10", min_value=0, step=1, value=0, key="qtd_m_10")
    qtd_05  = st.number_input("Qtd 0,05", min_value=0, step=1, value=0, key="qtd_m_05")
    qtd_01  = st.number_input("Qtd 0,01", min_value=0, step=1, value=0, key="qtd_m_01")
    total_moedas = (
        qtd_1*1.00 + qtd_50*0.50 + qtd_25*0.25 + qtd_10*0.10 + qtd_05*0.05 + qtd_01*0.01
    )
    st.metric("Total moedas", f"R$ {formatar_brasileiro(total_moedas)}")

# ---- Dinheiro
with col_dinheiro:
    st.subheader("💵 Dinheiro (físico)")
    q200 = st.number_input("Qtd 200", min_value=0, step=1, value=0, key="qtd_d_200")
    q100 = st.number_input("Qtd 100", min_value=0, step=1, value=0, key="qtd_d_100")
    q50  = st.number_input("Qtd 50",  min_value=0, step=1, value=0, key="qtd_d_50")
    q20  = st.number_input("Qtd 20",  min_value=0, step=1, value=0, key="qtd_d_20")
    q10  = st.number_input("Qtd 10",  min_value=0, step=1, value=0, key="qtd_d_10")
    q5   = st.number_input("Qtd 5",   min_value=0, step=1, value=0, key="qtd_d_5")
    q2   = st.number_input("Qtd 2",   min_value=0, step=1, value=0, key="qtd_d_2")
    total_dinheiro = (
        q200*200 + q100*100 + q50*50 + q20*20 + q10*10 + q5*5 + q2*2
    )
    st.metric("Total dinheiro", f"R$ {formatar_brasileiro(total_dinheiro)}")

# ---- Cartão
with col_cartao:
    st.subheader("💳 Cartão (físico)")
    cartao_fisico_txt = st.text_input("Valor (ex: 1.234,56)", value="0,00", key="cartao_fisico")
    cartao_fisico = br_to_float(cartao_fisico_txt)
    st.metric("Total cartão", f"R$ {formatar_brasileiro(cartao_fisico)}")

# ---- Pix (dinâmico)
with col_pix:
    st.subheader("📲 Pix (físico)")
    st.caption("Use a grade para inserir múltiplos lançamentos.")
    if "pix_df" not in st.session_state:
        st.session_state.pix_df = pd.DataFrame({"valor": ["0,00"]})
    pix_df = st.data_editor(
        st.session_state.pix_df,
        num_rows="dynamic",
        use_container_width=True,
        key="pix_editor",
    )
    st.session_state.pix_df = pix_df
# ----    total_pix_fisico = float(sum(br_to_float(v) for v in pix_df["valor"].tolist())) if not pix_df.empty else 0.0
    valores_pix = pix_df["valor"] if ("valor" in pix_df.columns) else []
    total_pix_fisico = float(
    pd.Series(valores_pix)
      .fillna("0")
      .astype(str)
      .map(br_to_float)
      .sum()
)
    st.metric("Total pix", f"R$ {formatar_brasileiro(total_pix_fisico)}")

# ---- Sangria (dinâmico)
with col_sangria:
    st.subheader("🏧 Sangria (físico)")
    if "sangria_df" not in st.session_state:
        st.session_state.sangria_df = pd.DataFrame({"valor": ["0,00"]})
    sangria_df = st.data_editor(
        st.session_state.sangria_df,
        num_rows="dynamic",
        use_container_width=True,
        key="sangria_editor",
    )
    st.session_state.sangria_df = sangria_df
# ----    total_sangria = float(sum(br_to_float(v) for v in sangria_df["valor"].tolist())) if not sangria_df.empty else 0.0
    valores_sangria = sangria_df["valor"] if ("valor" in sangria_df.columns) else []
    total_sangria = float(
    pd.Series(valores_sangria)
      .fillna("0")
      .astype(str)
      .map(br_to_float)
      .sum()
)
    st.metric("Total sangria", f"R$ {formatar_brasileiro(total_sangria)}")

# ---- Entrega (dinâmico)
with col_entrega:
    st.subheader("🛵 Entrega (físico)")
    if "entrega_df" not in st.session_state:
        st.session_state.entrega_df = pd.DataFrame({"valor": ["0,00"]})
    entrega_df = st.data_editor(
        st.session_state.entrega_df,
        num_rows="dynamic",
        use_container_width=True,
        key="entrega_editor",
    )
    st.session_state.entrega_df = entrega_df
# ----    total_entrega = float(sum(br_to_float(v) for v in entrega_df["valor"].tolist())) if not entrega_df.empty else 0.0
    valores_entrega = entrega_df["valor"] if ("valor" in entrega_df.columns) else []
    total_entrega = float(
    pd.Series(valores_entrega)
      .fillna("0")
      .astype(str)
      .map(br_to_float)
      .sum()
)
    st.metric("Total entrega", f"R$ {formatar_brasileiro(total_entrega)}")

# ---- Sistema + Resumo + Salvar
with col_sistema:
    st.subheader("🧾 Dados do Sistema + Resumo")
    if not st.session_state.operadores:
        st.error("⚠️ Carregue operadores.csv na barra lateral para liberar o salvamento.")
    nome_operador = st.selectbox("Operador", options=[""] + st.session_state.operadores, index=0)

    c1, c2 = st.columns(2)
    with c1:
        sys_dinheiro_txt = st.text_input("Sistema — Dinheiro", "0,00", key="sys_dinheiro")
        sys_debito_txt   = st.text_input("Sistema — Débito",   "0,00", key="sys_debito")
        sys_credito_txt  = st.text_input("Sistema — Crédito",  "0,00", key="sys_credito")
        sys_parc_txt     = st.text_input("Sistema — Parcelado","0,00", key="sys_parcelado")
    with c2:
        sys_qr_txt       = st.text_input("Sistema — Pix QR",   "0,00", key="sys_qr")
        sys_cora_txt     = st.text_input("Sistema — Pix Cora", "0,00", key="sys_cora")
        sys_entrega_txt  = st.text_input("Sistema — Entrega",  "0,00", key="sys_entrega")
        sys_troco_txt    = st.text_input("Sistema — Troco",    "0,00", key="sys_troco")

    # Converte sistema
    sys_dinheiro = br_to_float(sys_dinheiro_txt)
    sys_cartao   = br_to_float(sys_debito_txt) + br_to_float(sys_credito_txt) + br_to_float(sys_parc_txt)
    sys_pix      = br_to_float(sys_qr_txt) + br_to_float(sys_cora_txt)
    sys_entrega  = br_to_float(sys_entrega_txt)
    sys_troco    = br_to_float(sys_troco_txt)

    # Resumo (mesma regra do seu Tkinter)
    dinheiro_fisico = total_moedas + total_dinheiro
    dif_dinheiro = dinheiro_fisico - sys_dinheiro
    dif_cartao   = cartao_fisico - sys_cartao
    dif_pix      = total_pix_fisico - sys_pix
    dif_entrega  = total_entrega - sys_entrega
    total_geral  = dif_dinheiro + dif_cartao + dif_pix + dif_entrega

    st.divider()
    st.markdown("**Resumo (Físico vs Sistema)**")
    resumo_df = pd.DataFrame(
        [
            ["Dinheiro", dinheiro_fisico, sys_dinheiro, dif_dinheiro],
            ["Cartão",   cartao_fisico,   sys_cartao,   dif_cartao],
            ["Pix",      total_pix_fisico, sys_pix,     dif_pix],
            ["Entrega",  total_entrega,   sys_entrega,  dif_entrega],
        ],
        columns=["Tipo", "Físico", "Sistema", "Diferença"],
    )
    # formata para BR
    resumo_df_fmt = resumo_df.copy()
    for col in ["Físico", "Sistema", "Diferença"]:
        resumo_df_fmt[col] = resumo_df_fmt[col].apply(lambda x: f"R$ {formatar_brasileiro(float(x))}")
    st.dataframe(resumo_df_fmt, use_container_width=True, hide_index=True)

    st.metric("TOTAL GERAL (diferenças)", f"R$ {formatar_brasileiro(total_geral)}")

    st.divider()
    cbtn1, cbtn2 = st.columns([1, 2])
    with cbtn1:
        salvar_click = st.button("💾 Salvar fechamento", use_container_width=True)
    with cbtn2:
        st.caption("Salva por data + operador (substitui se já existir).")

    if salvar_click:
        if not nome_operador.strip():
            st.error("Selecione um operador.")
        else:
            # Mesma regra: exige ter pelo menos algum valor no sistema e total_geral calculável
            if all(x == 0.0 for x in [sys_dinheiro, sys_cartao, sys_pix, sys_entrega, sys_troco]):
                st.error("Preencha os campos do SISTEMA antes de salvar.")
            else:
                payload = dict(
                    data=data_sqlite,
                    nome_operador=nome_operador.strip(),
                    moedas_fisico=float(total_moedas),
                    dinheiro_fisico=float(total_dinheiro),
                    cartao_fisico=float(cartao_fisico),
                    pix_fisico=float(total_pix_fisico),
                    entrega_fisico=float(total_entrega),
                    sangria_fisico=float(total_sangria),
                    sistema_dinheiro=float(sys_dinheiro),
                    sistema_cartao=float(sys_cartao),
                    sistema_pix=float(sys_pix),
                    sistema_entrega=float(sys_entrega),
                    sistema_troco=float(sys_troco),
                    dif_dinheiro=float(dif_dinheiro),
                    dif_cartao=float(dif_cartao),
                    dif_pix=float(dif_pix),
                    dif_entrega=float(dif_entrega),
                    total_geral=float(total_geral),
                )
                salvar_fechamento(payload)
                st.success(f"Fechamento salvo para {nome_operador} em {data_sqlite}.")

# ==============================
# Histórico (se carregado)
# ==============================
if "_hist" in st.session_state:
    st.subheader("📚 Histórico")
    st.dataframe(st.session_state._hist, use_container_width=True)
    st.download_button(
        "⬇️ Baixar histórico (CSV)",
        data=st.session_state._hist.to_csv(index=False).encode("utf-8"),
        file_name="fechamentos.csv",
        mime="text/csv",
    )
