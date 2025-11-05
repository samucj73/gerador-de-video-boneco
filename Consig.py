# app.py
import streamlit as st
import numpy as np
import pandas as pd
import math
from io import StringIO

st.set_page_config(page_title="Renovar Consignado — Simulador", layout="wide")

# ----- CSS / Header -----
st.markdown("""
<style>
.header {
  background: linear-gradient(90deg, #0b6cff, #6b00ff);
  padding: 18px;
  border-radius: 8px;
  color: white;
  font-family: "Segoe UI", Roboto, sans-serif;
}
.card {
  background: #fff;
  padding: 14px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(11,20,60,0.06);
}
.small {
  font-size:13px; color:#666;
}
.result-box {
  background: linear-gradient(180deg,#f8fbff,#ffffff);
  border-left: 6px solid #0b6cff;
  padding: 14px;
  border-radius: 6px;
}
</style>
<div class="header">
  <h2 style="margin:0">Simulador de Renovação de Empréstimo Consignado</h2>
  <div class="small">Calcule troco, reduções de parcela ou redução de prazo ao renovar seu consignado</div>
</div>
""", unsafe_allow_html=True)

st.write(" ")

# ----- Sidebar: inputs -----
st.sidebar.header("Parâmetros do contrato atual")

# Novos inputs requisitados pelo usuário
total_parcelas_orig = st.sidebar.number_input("Total de parcelas do contrato original", min_value=1, value=60, step=1)
parcelas_pagas = st.sidebar.number_input("Parcelas já pagas", min_value=0, value=42, step=1)
# garante consistência
if parcelas_pagas > total_parcelas_orig:
    st.sidebar.error("Parcelas já pagas não pode ser maior que o total de parcelas do contrato original.")
parcelas_restantes_builtin = max(0, int(total_parcelas_orig - parcelas_pagas))
st.sidebar.write(f"Parcelas restantes (calculadas): {parcelas_restantes_builtin}")

input_mode = st.sidebar.radio("Como informar o saldo atual?", ("Informar Saldo Atual", "Calcular do pagamento atual"))

if input_mode == "Informar Saldo Atual":
    saldo_atual = st.sidebar.number_input("Saldo devedor atual (R$)", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
    parcela_atual = st.sidebar.number_input("Parcela atual (opcional, R$)", min_value=0.0, value=350.0, step=10.0, format="%.2f")
    # se o usuário preferir, pode usar as parcelas calculadas acima
    parcelas_restantes = st.sidebar.number_input("Parcelas restantes (opcional)", min_value=0, value=parcelas_restantes_builtin, step=1)
else:
    parcela_atual = st.sidebar.number_input("Parcela atual (R$)", min_value=1.0, value=350.0, step=10.0, format="%.2f")
    # usa parcelas restantes calculadas do contrato original por padrão
    parcelas_restantes = st.sidebar.number_input("Parcelas restantes", min_value=1, value=parcelas_restantes_builtin if parcelas_restantes_builtin>0 else 18, step=1)
    saldo_atual = None

st.sidebar.header("Condições do refinanciamento (novo contrato)")
taxa_anuaria = st.sidebar.number_input("Taxa anual do novo contrato (%)", min_value=0.0, value=26.0, step=0.1, format="%.2f")
taxa_mensal = (1 + taxa_anuaria/100)**(1/12) - 1  # convert anual efetiva para mensal aproximada
st.sidebar.write(f"Taxa mensal equivalente: {taxa_mensal*100:.3f}%")

opcao = st.sidebar.selectbox("Objetivo da renovação", ["Pegar troco (cashback)", "Diminuir parcela mantendo prazo", "Diminuir prazo mantendo parcela"])
st.sidebar.write("Taxas e tarifas (opcional)")
tarifa = st.sidebar.number_input("Tarifa administrativa/IOF (R$)", min_value=0.0, value=0.0, step=1.0, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.write("Exportar / copiar resultados ao final")

# ----- Main: scenario inputs -----
st.markdown("## Dados da Simulação")

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Saldo atual / origem")
    if input_mode == "Informar Saldo Atual":
        st.write("Você informou o saldo atual diretamente.")
        st.write(f"**Saldo atual:** R$ {saldo_atual:,.2f}")
    else:
        st.write("Saldo será calculado a partir da parcela atual e parcelas restantes (presupõe-se amortização padrão).")
        st.write(f"**Parcela atual:** R$ {parcela_atual:,.2f}")
        st.write(f"**Parcelas restantes (usadas):** {int(parcelas_restantes)}")

with col2:
    st.subheader("Escolha do que deseja")
    if opcao == "Pegar troco (cashback)":
        # NOVA LOGICA: modo de troco
        modo_troco = st.radio("Modo de troco:", ("Calcular troco máximo mantendo parcela atual", "Informar troco manualmente"))
        if modo_troco == "Informar troco manualmente":
            troco_desejado = st.number_input("Valor de troco desejado (R$)", min_value=0.0, value=1000.0, step=50.0, format="%.2f")
            novo_prazo = st.number_input("Novo prazo (meses) — opcional", min_value=1, value=36, step=1)
            manter_parcela = None
        else:
            # cálculo do troco máximo
            novo_prazo = st.number_input("Novo prazo (meses) para o novo contrato (usado no cálculo do troco máximo)", min_value=1, value=36, step=1)
            manter_parcela = st.number_input("Parcela atual a ser mantida (R$) — normalmente sua parcela atual", min_value=1.0, value=parcela_atual if parcela_atual>0 else 350.0, step=10.0, format="%.2f")
            troco_desejado = None
    elif opcao == "Diminuir parcela mantendo prazo":
        troco_desejado = st.number_input("Troco (se quiser, R$)", min_value=0.0, value=0.0, step=50.0, format="%.2f")
        novo_prazo = st.number_input("Novo prazo (meses)", min_value=1, value=parcelas_restantes if parcelas_restantes>0 else 36, step=1)
        manter_parcela = None
    else:  # Diminuir prazo mantendo parcela
        manter_parcela = st.number_input("Parcela desejada (R$)", min_value=1.0, value=parcela_atual if parcela_atual>0 else 350.0, step=10.0, format="%.2f")
        troco_desejado = st.number_input("Troco (se quiser, R$)", min_value=0.0, value=0.0, step=50.0, format="%.2f")
        novo_prazo = None

st.write(" ")

# ----- Helper functions -----
def pv_of_annuity(pmt, i, n):
    """Valor presente (saldo) de n parcelas pmt com taxa mensal i."""
    if i == 0:
        return pmt * n
    return pmt * (1 - (1 + i) ** (-n)) / i

def pmt_from_pv(pv, i, n):
    """Parcela mensal para PV, taxa i mensal, n meses."""
    if n == 0:
        return float('inf')
    if i == 0:
        return pv / n
    return pv * i / (1 - (1 + i) ** (-n))

def n_from_pv_pmt(pv, pmt, i):
    """Número de parcelas necessárias para pagar pv com pmt e taxa i."""
    if pmt <= pv * i:
        return None  # não é suficiente para cobrir juros
    if i == 0:
        return pv / pmt
    try:
        n = -math.log(1 - pv * i / pmt) / math.log(1 + i)
        return n
    except:
        return None

# ----- Compute saldo atual if needed -----
if input_mode == "Calcular do pagamento atual":
    if parcela_atual <= 0 or parcelas_restantes <= 0:
        st.error("Informe parcela atual e parcelas restantes corretamente.")
        st.stop()
    saldo_atual = pv_of_annuity(parcela_atual, taxa_mensal, parcelas_restantes)

saldo_atual = float(saldo_atual)

# ----- Lógica especial para 'Pegar troco' com modo calcular troco máximo -----
troco = 0.0
novo_principal = None

if opcao == "Pegar troco (cashback)" and 'modo_troco' in locals() and modo_troco == "Calcular troco máximo mantendo parcela atual":
    # queremos calcular troco_max tal que: PV_novo = PV_max = parcela_mantida * (1 - (1+i)^-n) / i
    # então troco_max = PV_max - saldo_atual - tarifa
    PV_max = pv_of_annuity(manter_parcela, taxa_mensal, int(novo_prazo))
    troco_max = PV_max - saldo_atual - tarifa
    # troco não pode ser negativo: se negativo → não tem troco possível mantendo parcela
    if troco_max < 0:
        troco_max = 0.0
        aviso_troco = "Nenhum troco disponível mantendo a parcela atual (ou parcela atual é muito baixa para este prazo/taxa)."
    else:
        aviso_troco = None
    troco = troco_max
    novo_principal = saldo_atual + troco + tarifa
elif opcao == "Pegar troco (cashback)" and 'modo_troco' in locals() and modo_troco == "Informar troco manualmente":
    troco = float(troco_desejado)
    novo_principal = saldo_atual + troco + tarifa
else:
    # outros cenários: troco informado ou zero
    troco = float(troco_desejado) if 'troco_desejado' in locals() and troco_desejado is not None else 0.0
    novo_principal = saldo_atual + troco + tarifa

# ----- Calculations by scenario -----
st.markdown("## Resultado da Simulação")
resumo = {}

if opcao == "Pegar troco (cashback)":
    if 'modo_troco' in locals() and modo_troco == "Calcular troco máximo mantendo parcela atual":
        n = int(novo_prazo)
        nova_parcela = pmt_from_pv(novo_principal, taxa_mensal, n)
        resumo['Objetivo'] = 'Troco (calculado mantendo parcela atual)'
        resumo['Saldo atual'] = saldo_atual
        resumo['Parcela mantida (R$)'] = manter_parcela
        resumo['Troco calculado (R$)'] = troco
        if aviso_troco:
            resumo['Aviso troco'] = aviso_troco
        resumo['Tarifa/IOF'] = tarifa
        resumo['Novo principal (saldo+troco+tarifa)'] = novo_principal
        resumo['Novo prazo (meses)'] = n
        resumo['Nova parcela (R$)'] = nova_parcela
        resumo['Taxa anual (%)'] = taxa_anuaria
        resumo['Taxa mensal (%)'] = taxa_mensal*100
    else:
        # troco manual
        n = int(novo_prazo)
        nova_parcela = pmt_from_pv(novo_principal, taxa_mensal, n)
        resumo['Objetivo'] = 'Troco (manual)'
        resumo['Saldo atual'] = saldo_atual
        resumo['Troco desejado'] = troco
        resumo['Tarifa/IOF'] = tarifa
        resumo['Novo principal (saldo+troco+tarifa)'] = novo_principal
        resumo['Novo prazo (meses)'] = n
        resumo['Nova parcela (R$)'] = nova_parcela
        resumo['Taxa anual (%)'] = taxa_anuaria
        resumo['Taxa mensal (%)'] = taxa_mensal*100

elif opcao == "Diminuir parcela mantendo prazo":
    n = int(novo_prazo)
    nova_parcela = pmt_from_pv(novo_principal, taxa_mensal, n)
    resumo['Objetivo'] = 'Diminuir parcela (mesmo prazo)'
    resumo['Saldo atual'] = saldo_atual
    resumo['Troco (R$)'] = troco
    resumo['Tarifa/IOF'] = tarifa
    resumo['Novo principal'] = novo_principal
    resumo['Prazo (meses)'] = n
    resumo['Nova parcela (R$)'] = nova_parcela
    resumo['Taxa anual (%)'] = taxa_anuaria
    resumo['Taxa mensal (%)'] = taxa_mensal*100

else:  # Diminuir prazo mantendo parcela
    pmt = manter_parcela
    resumo['Objetivo'] = 'Diminuir prazo (mesma parcela)'
    resumo['Saldo atual'] = saldo_atual
    resumo['Troco (R$)'] = troco
    resumo['Tarifa/IOF'] = tarifa
    resumo['Novo principal'] = novo_principal
    resumo['Parcela mantida (R$)'] = pmt
    resumo['Taxa anual (%)'] = taxa_anuaria
    resumo['Taxa mensal (%)'] = taxa_mensal*100

    n_calc = n_from_pv_pmt(novo_principal, pmt, taxa_mensal)
    if n_calc is None:
        resumo['Obs'] = "A parcela informada não cobre os juros: é insuficiente para amortizar a dívida."
        prazo_mes = None
    else:
        prazo_mes = math.ceil(n_calc)
        resumo['Novo prazo estimado (meses)'] = n_calc
        resumo['Novo prazo arredondado (meses)'] = prazo_mes
        pmt_real = pmt_from_pv(novo_principal, taxa_mensal, prazo_mes)
        resumo['Parcela efetiva para prazo arredondado (R$)'] = pmt_real

# ----- Mostrar resultados -----
st.write("")
left, right = st.columns([2,1])
with left:
    st.subheader("Resumo rápido")
    for k,v in resumo.items():
        if isinstance(v, (int, float)):
            if any(word in k.lower() for word in ["saldo","troco","tarifa","parcela","principal"]):
                st.write(f"**{k}:** R$ {float(v):,.2f}")
            elif "taxa" in k.lower():
                st.write(f"**{k}:** {float(v):.4f}" + ("%"))
            else:
                st.write(f"**{k}:** {v}")
        else:
            st.write(f"**{k}:** {v}")

with right:
    st.subheader("Detalhes / Números")
    st.table(pd.DataFrame([
        ["Total parcelas originais", f"{int(total_parcelas_orig)}"],
        ["Parcelas já pagas", f"{int(parcelas_pagas)}"],
        ["Parcelas restantes (calculadas)", f"{int(parcelas_restantes_builtin)}"],
        ["Saldo atual (R$)", f"{saldo_atual:,.2f}"],
        ["Troco (R$)", f"{troco:,.2f}"],
        ["Tarifa (R$)", f"{tarifa:,.2f}"],
        ["Novo principal (R$)", f"{novo_principal:,.2f}"],
        ["Taxa anual (%)", f"{taxa_anuaria:.3f}"],
        ["Taxa mensal (%)", f"{taxa_mensal*100:.4f}"]
    ], columns=["Descrição", "Valor"]).set_index("Descrição"))

st.write("---")

# ----- Amortization sample table (primeiras 24 meses) -----
st.subheader("Projeção de amortização (primeiros 24 meses do novo contrato)")

if opcao == "Diminuir prazo mantendo parcela" and resumo.get('Obs') is not None:
    st.error(resumo['Obs'])
else:
    if opcao == "Diminuir prazo mantendo parcela":
        if resumo.get('Novo prazo arredondado (meses)') is not None:
            n_display = int(resumo['Novo prazo arredondado (meses)'])
            pmt_use = manter_parcela
        else:
            n_display = 24
            pmt_use = manter_parcela
    else:
        n_display = int(novo_prazo) if novo_prazo is not None else 24
        pmt_use = pmt_from_pv(novo_principal, taxa_mensal, n_display)

    rows = []
    bal = novo_principal
    months = min(n_display, 24)
    for m in range(1, months+1):
        juros = bal * taxa_mensal
        amort = pmt_use - juros
        if amort < 0:
            amort = 0
        bal_next = bal - amort
        rows.append({
            "Mês": m,
            "Parcela (R$)": round(pmt_use,2),
            "Juros (R$)": round(juros,2),
            "Amortização (R$)": round(amort,2),
            "Saldo após pagamento (R$)": round(bal_next if bal_next>0 else 0.0,2)
        })
        bal = bal_next
        if bal <= 0:
            break

    am_df = pd.DataFrame(rows)
    st.dataframe(am_df, height=300)

# ----- Export: gerar CSV ou copiar texto -----
st.write("---")
st.subheader("Exportar resultados")

csv_buffer = StringIO()
out_df = pd.DataFrame([
    {"chave": k, "valor": (v if not isinstance(v, float) else round(v,4))} for k,v in resumo.items()
])
out_df.to_csv(csv_buffer, index=False, sep=';')
csv_data = csv_buffer.getvalue().encode('utf-8')

st.download_button("Baixar resumo (CSV)", data=csv_data, file_name="simulacao_renovacao.csv", mime="text/csv")

txt_summary = "\n".join([f"{k}: {v}" for k,v in resumo.items()])
st.download_button("Baixar resumo (TXT)", data=txt_summary, file_name="simulacao_renovacao.txt", mime="text/plain")

st.markdown("""
<small class="small">Observações: 
<ul>
<li>Fórmulas: parcela = PV * i / (1-(1+i)^-n). Saldo presente de uma anuidade = parcela * (1-(1+i)^-n)/i.</li>
<li>Taxa anual convertida para taxa mensal por: (1+ja)^(1/12)-1 (aproximação com capitalização mensal).</li>
<li>Resultados são estimativas — bancos podem aplicar tarifas, arredondamentos, seguro, IOF ou regras contratuais.</li>
</ul>
</small>
""", unsafe_allow_html=True)
