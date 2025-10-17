        time_field = e.get("event_time") or e.get("time") or e.get("strTime") or e.get("timeEvent")
        event_id league_identifier: str) -> Dict[str, Any]:
    """
    Tenta obter standings para uma liga. league_identifier pode ser id/slug/nome.
            st.session_state.data_ultima_busca = None
        st.session_state.resultados_conferidos = []
        st.success("Dados da sessão limpos!")
        st.rerun()

    st.markdown("---")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        buscar_btn = st.button("🔍 Buscar partidas e analisar", type="primary")
    with c2:
        enviar_alertas_btn = st.button("🚀 Enviar Alertas Individuais", disabled=not st.session_state.busca_realizada)
    with c3:
        enviar_top_btn = st.button("📊 Enviar Top Consolidado", disabled=not st.session_state.busca_realizada)

    # BUSCA
    if buscar_btn:
        with st.spinner("Buscando partidas e analisando (AllSportsAPI)..."):
            jogos_encontrados, top_jogos, alertas_auto_enviados = buscar_e_analisar_jogos_allsports(
                data_selecionada, ligas_selecionadas, ligas_selecionadas_ids, conf_threshold, auto_send
            )
            st.session_state.jogos_encontrados = jogos_encontrados
            st.session_state.top_jogos = top_jogos
            st.session_state.busca_realizada = True
            st.session_state.data_ultima_busca = data_str
            st.session_state.alertas_enviados = len(alertas_auto_enviados) > 0

        if jogos_encontrados:
            st.success(f"✅ {len(jogos_encontrados)} jogos encontrados e analisados!")
            if alertas_auto_enviados:
                st.success(f"📨 {len(alertas_auto_enviados)} alertas automáticos enviados (conf >= {conf_threshold}%)")
            st.subheader("📋 Todos os Jogos Encontrados")
            for jogo in jogos_encontrados:
                with st.container():
                    col_a, col_b, col_c = st.columns([3,2,1])
                    with col_a:
                        st.write(f"**{jogo['home']}** vs **{jogo['away']}**")
                        st.write(f"🏆 {jogo['liga']} | 🕐 {jogo['hora']} | 📊 {jogo['origem']}")
                    with col_b:
                        st.write(f"🎯 {jogo['tendencia']}")
                        st.write(f"📈 Estimativa: {jogo['estimativa']} | ✅ Confiança: {jogo['confianca']}%")
                    with col_c:
                        if jogo in st.session_state.top_jogos:
                            st.success("🏆 TOP")
                    st.divider()
            if top_jogos:
                st.subheader("🏆 Top 5 Jogos (Maior Confiança)")
                for i, jogo in enumerate(top_jogos, 1):
                    st.info(f"{i}. **{jogo['home']}** vs **{jogo['away']}** - {jogo['tendencia']} ({jogo['confianca']}% confiança)")
        else:
            st.warning("⚠️ Nenhum jogo encontrado para os critérios selecionados.")

    # Envio manual de alertas individuais
    if enviar_alertas_btn and st.session_state.busca_realizada:
        with st.spinner("Enviando alertas individuais..."):
            alertas_enviados = enviar_alertas_individualmente(st.session_state.jogos_encontrados)
            if alertas_enviados:
                st.session_state.alertas_enviados = True
                st.success(f"✅ {len(alertas_enviados)} alertas enviados com sucesso!")
            else:
                st.error("❌ Erro ao enviar alertas (ou nenhum alerta enviado)")

    # Envio Top consolidado
    if enviar_top_btn and st.session_state.busca_realizada and st.session_state.top_jogos:
        with st.spinner("Enviando top consolidado..."):
            if enviar_top_consolidado(st.session_state.top_jogos):
                st.success("✅ Top consolidado enviado com sucesso!")
            else:
                st.error("❌ Erro ao enviar top consolidado")

    # Conferência de resultados (esqueleto)
    st.markdown("---")
    conferir_btn = st.button("📊 Conferir resultados (usar alertas salvo)")
    if conferir_btn:
        st.info("Conferindo resultados dos alertas salvos... (implemente sua lógica de conferência aqui)")
        # Exemplo: carregar alertas e verificar via
