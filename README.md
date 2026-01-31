# Caixa - Fechamento Online (Streamlit)

## Rodar local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Operadores
- Crie um `operadores.csv` (UTF-8) com **uma coluna** chamada `nome`.
- Você também pode enviar o arquivo pela barra lateral do app (ele será salvo no servidor).

## Deploy (Streamlit Community Cloud)
- Suba `app.py`, `requirements.txt` e (opcional) `operadores.csv` num repo.
- No Community Cloud, selecione o repo e aponte para `app.py`.

> Observação: este deploy com SQLite atende bem 1 loja / baixo volume.
> Para multi-lojas, concorrência e auditoria, migre para Postgres + backend (FastAPI).
