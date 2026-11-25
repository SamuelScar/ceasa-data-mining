"""Exibe o andamento dos pipelines sem perder a saída das etapas."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from time import perf_counter

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


def duracao_legivel(segundos: float) -> str:
    total = round(segundos)
    horas, resto = divmod(total, 3600)
    minutos, segundos_restantes = divmod(resto, 60)
    if horas:
        return f"{horas}h {minutos:02d}min {segundos_restantes:02d}s"
    if minutos:
        return f"{minutos}min {segundos_restantes:02d}s"
    return f"{segundos_restantes}s"


class PainelPipeline:
    """Mantém o progresso no rodapé e os logs no histórico do terminal."""

    def __init__(self, titulo: str, total_etapas: int) -> None:
        self.titulo = titulo
        self.total_etapas = total_etapas
        self.console = Console()
        self.interativo = self.console.is_terminal
        self.inicio = perf_counter()
        self.concluidas = 0
        self.indice_atual = 0
        self.etapa_atual = "Preparando execução"
        self.ultima_mensagem = "Aguardando a primeira etapa"
        self.status = "aguardando"
        self.live: Live | None = None

    def __enter__(self) -> PainelPipeline:
        if self.interativo:
            self.live = Live(
                console=self.console,
                get_renderable=self._renderizar,
                refresh_per_second=4,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self.live.start(refresh=True)
        else:
            self._exibir_estado_simples()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc is not None and self.status != "falha":
            self.status = "falha"
            self.ultima_mensagem = str(exc)
        if self.live is not None:
            self.live.stop()

    def exibir_log(self, mensagem: str = "") -> None:
        self.console.print(Text(mensagem), highlight=False)

    def iniciar_etapa(self, indice: int, nome: str, descricao: str) -> None:
        self.indice_atual = indice
        self.etapa_atual = nome
        self.ultima_mensagem = descricao
        self.status = "executando"
        self.exibir_log()
        self.exibir_log(f">>> Etapa {indice}/{self.total_etapas}: {nome}")
        self.exibir_log(f"    {descricao}")

    def registrar_saida(self, linha: str) -> None:
        mensagem = linha.rstrip("\r\n")
        if mensagem:
            self.ultima_mensagem = mensagem
        self.exibir_log(mensagem)

    def concluir_etapa(self, indice: int, nome: str) -> None:
        self.concluidas = indice
        self.ultima_mensagem = f"Etapa concluída: {nome}"
        self.status = "aguardando"
        if not self.interativo:
            self._exibir_estado_simples()

    def iniciar_finalizacao(self) -> None:
        self.etapa_atual = "Finalizando artefatos"
        self.ultima_mensagem = "Calculando hashes e concluindo o manifesto"
        self.status = "executando"

    def falhar_etapa(self, nome: str, codigo_saida: int) -> None:
        self.status = "falha"
        self.ultima_mensagem = (
            f"Falha em {nome} com código de saída {codigo_saida}"
        )
        if not self.interativo:
            self._exibir_estado_simples()

    def finalizar(self) -> None:
        self.concluidas = self.total_etapas
        self.indice_atual = self.total_etapas
        self.etapa_atual = "Todas as etapas concluídas"
        self.ultima_mensagem = "Pipeline concluído com sucesso"
        self.status = "concluido"
        if self.live is not None:
            self.live.refresh()

    def _renderizar(self) -> Panel:
        tabela = Table.grid(expand=True, padding=(0, 1))
        tabela.add_column(style="bold", width=7, no_wrap=True)
        tabela.add_column(ratio=1)
        tabela.add_column(justify="right", no_wrap=True)

        percentual = 100 * self.concluidas / self.total_etapas
        tabela.add_row(
            "Geral",
            ProgressBar(
                total=self.total_etapas,
                completed=self.concluidas,
                width=None,
            ),
            f"{self.concluidas}/{self.total_etapas}  {percentual:3.0f}%",
        )
        tabela.add_row(
            "Etapa",
            Text(self.etapa_atual, overflow="ellipsis", no_wrap=True),
            f"{self.indice_atual}/{self.total_etapas}",
        )
        tabela.add_row("Status", self._renderizar_status(), "")
        tabela.add_row(
            "Atual",
            Text(self.ultima_mensagem, overflow="ellipsis", no_wrap=True),
            "",
        )
        return Panel(
            tabela,
            title=self.titulo,
            border_style=self._cor_status(),
        )

    def _renderizar_status(self) -> Text | Spinner:
        tempo = duracao_legivel(perf_counter() - self.inicio)
        if self.status == "executando":
            return Spinner("dots", text=f"Em execução há {tempo}")
        if self.status == "concluido":
            return Text(f"Concluído em {tempo}", style="green")
        if self.status == "falha":
            return Text(f"Falhou após {tempo}", style="bold red")
        return Text(f"Aguardando | tempo total: {tempo}", style="yellow")

    def _cor_status(self) -> str:
        if self.status == "concluido":
            return "green"
        if self.status == "falha":
            return "red"
        return "cyan"

    def _exibir_estado_simples(self) -> None:
        percentual = 100 * self.concluidas / self.total_etapas
        self.exibir_log(
            f"Progresso geral: {self.concluidas}/{self.total_etapas} "
            f"({percentual:.2f}%) | "
            f"Tempo: {duracao_legivel(perf_counter() - self.inicio)}"
        )


def executar_comando(comando: Sequence[str], painel: PainelPipeline) -> int:
    """Executa um processo e encaminha sua saída ao painel em tempo real."""
    with subprocess.Popen(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as processo:
        assert processo.stdout is not None
        for linha in processo.stdout:
            painel.registrar_saida(linha)
        return processo.wait()
