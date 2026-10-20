"use server";

import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";
import { db, ler, listar } from "@/lib/db";
import { falha, sucesso, type Resultado } from "@/lib/resultado";
import { inscreverProfessores } from "@/lib/inscricoes";
import { atualizarPeriodos, calendarioDoAno, MAX_PERIODOS, periodicidadeDe, periodosEmCurso, prazoFinal } from "@/lib/recorrencia";
import { anoAtual, anoDaCampanha } from "@/lib/ano";
import type { Campanha, DadosPagamento, Inscricao, Instituicao, Professor } from "@/lib/tipos";
import {
  autenticar,
  criarSessao,
  encerrarSessao,
  hashSenha,
  requireInstituicaoAcao,
  requireSuper,
} from "@/lib/auth";
import { fimDoDia, fmtData, normalizarTelefone } from "@/lib/format";
import { enviarLembretesSelecionados, executarLembretes } from "@/lib/lembretes";
import { enviarSmsAosProfessores } from "@/lib/mensagens";
import { smsConfigurado } from "@/lib/sms";
import { MAX_SEGMENTOS, MAX_SEPTETOS, paraSms, septetos } from "@/lib/smsTexto";
import { canalValido, smsAtivo } from "@/lib/canais";
import {
  ibanValido,
  multicaixaValido,
  normalizarMulticaixa,
} from "@/lib/dadosPagamento";
import {
  alterarValorDevido,
  anularPagamentoManual,
  apagarComprovativo,
  aprovarComprovativo,
  corrigirValorAprovado,
  registarPagamentoManual,
  rejeitarComprovativo,
} from "@/lib/pagamentos";
import {
  apagarCampanhaCompleta,
  apagarInstituicaoCompleta,
  apagarProfessorCompleto,
  removerInscricaoCompleta,
  removerInscricoesCompleto,
} from "@/lib/apagar";

/** Destino opcional (campo "ir") para onde o navegador vai depois de apagar: só caminhos internos. */
const destino = (fd: FormData): string | undefined => {
  const v = String(fd.get("ir") ?? "");
  return /^\/[a-z0-9/_-]*$/i.test(v) && !v.startsWith("//") ? v : undefined;
};
const texto = (fd: FormData, k: string) => String(fd.get(k) ?? "").trim();
const inteiro = (fd: FormData, k: string) => parseInt(texto(fd, k).replace(/[^\d]/g, ""), 10);

// ---------- Sessão ----------

export async function loginAction(fd: FormData) {
  const sessao = await autenticar(texto(fd, "email"), String(fd.get("senha") ?? ""));
  if (!sessao) redirect("/login?erro=1");
  await criarSessao(sessao);
  redirect(sessao.papel === "super" ? "/instituicoes" : "/");
}

export async function logoutAction() {
  await encerrarSessao();
  redirect("/login");
}

// ---------- Instituições (só o dono da plataforma) ----------

export async function criarInstituicao(fd: FormData): Promise<Resultado> {
  await requireSuper();
  const nome = texto(fd, "nome");
  const adminEmail = texto(fd, "adminEmail").toLowerCase();
  const senha = String(fd.get("senha") ?? "");
  if (!nome) return falha("Indica o nome da instituição.");
  if (!/^\S+@\S+\.\S+$/.test(adminEmail)) return falha("Indica um email válido para o admin.");
  if (senha.length < 8) return falha("A senha tem de ter pelo menos 8 caracteres.");
  if (adminEmail === (process.env.ADMIN_EMAIL ?? "").trim().toLowerCase()) {
    return falha("Esse email é o do dono da plataforma. Usa outro.");
  }
  const existe = await db().collection("instituicoes").where("adminEmail", "==", adminEmail).limit(1).get();
  if (!existe.empty) return falha("Já existe uma instituição com este email de admin.");

  const dados: Omit<Instituicao, "id"> = {
    nome,
    adminEmail,
    adminSenha: hashSenha(senha),
    tomCristao: Boolean(fd.get("tomCristao")),
    canal: canalValido(texto(fd, "canal")),
    ativa: true,
    criadaEm: new Date(),
  };
  await db().collection("instituicoes").add(dados);
  revalidatePath("/instituicoes");
  return sucesso(`${nome} criada.`);
}

export async function entrarInstituicao(fd: FormData) {
  await requireSuper();
  const id = texto(fd, "id");
  const i = ler<Instituicao>(await db().collection("instituicoes").doc(id).get());
  if (!i) redirect("/instituicoes");
  await criarSessao({ papel: "super", instituicaoId: i.id });
  redirect("/");
}

export async function sairInstituicao() {
  await requireSuper();
  await criarSessao({ papel: "super", instituicaoId: null });
  redirect("/instituicoes");
}

export async function alternarInstituicao(fd: FormData): Promise<Resultado> {
  await requireSuper();
  const ref = db().collection("instituicoes").doc(texto(fd, "id"));
  const i = ler<Instituicao>(await ref.get());
  if (!i) return falha("Instituição não encontrada.");
  await ref.update({ ativa: !i.ativa });
  revalidatePath("/instituicoes");
  return sucesso(i.ativa ? `${i.nome} desativada.` : `${i.nome} ativada.`);
}

export async function redefinirSenhaInstituicao(fd: FormData): Promise<Resultado> {
  await requireSuper();
  const id = texto(fd, "id");
  const rota = `/instituicoes/${id}`;
  const senha = String(fd.get("senha") ?? "");
  if (senha.length < 8) return falha("A senha tem de ter pelo menos 8 caracteres.");
  const ref = db().collection("instituicoes").doc(id);
  if (!(await ref.get()).exists) return falha("Instituição não encontrada.");
  await ref.update({ adminSenha: hashSenha(senha) });
  return sucesso("Senha atualizada.");
}

export async function editarInstituicao(fd: FormData): Promise<Resultado> {
  await requireSuper();
  const id = texto(fd, "id");
  const rota = `/instituicoes/${id}`;
  const ref = db().collection("instituicoes").doc(id);
  if (!(await ref.get()).exists) return falha("Instituição não encontrada.");

  const nome = texto(fd, "nome");
  const adminEmail = texto(fd, "adminEmail").toLowerCase();
  if (!nome) return falha("Indica o nome da instituição.");
  if (!/^\S+@\S+\.\S+$/.test(adminEmail)) return falha("Indica um email válido para o admin.");
  if (adminEmail === (process.env.ADMIN_EMAIL ?? "").trim().toLowerCase()) {
    return falha("Esse email é o do dono da plataforma. Usa outro.");
  }
  const outro = await db().collection("instituicoes").where("adminEmail", "==", adminEmail).get();
  if (outro.docs.some((d) => d.id !== id)) return falha("Já existe outra instituição com este email de admin.");

  await ref.update({ nome, adminEmail, tomCristao: Boolean(fd.get("tomCristao")), canal: canalValido(texto(fd, "canal")) });
  revalidatePath("/instituicoes");
  revalidatePath(rota);
  revalidatePath("/", "layout"); // o menu mostra ou esconde a página SMS
  return sucesso("Instituição atualizada.");
}

/** Apaga a instituição e tudo o que é dela. Exige escrever o nome exato. */
export async function apagarInstituicao(fd: FormData): Promise<Resultado> {
  const sessao = await requireSuper();
  const id = texto(fd, "id");
  const rota = `/instituicoes/${id}`;
  const i = ler<Instituicao>(await db().collection("instituicoes").doc(id).get());
  if (!i) return falha("Instituição não encontrada.");
  if (texto(fd, "confirmacao") !== i.nome) return falha("Para apagar, escreve o nome exato da instituição.");

  await apagarInstituicaoCompleta(id);
  if (sessao.instituicaoId === id) await criarSessao({ papel: "super", instituicaoId: null });
  revalidatePath("/instituicoes");
  return sucesso(`${i.nome} e todos os seus dados foram apagados.`, "/instituicoes");
}

// ---------- Dados de pagamento da instituição ----------

export async function guardarDadosPagamento(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const rota = "/pagamento";
  const multicaixaNumero = normalizarMulticaixa(texto(fd, "multicaixaNumero"));
  const multicaixaTitular = texto(fd, "multicaixaTitular");
  const iban = texto(fd, "iban").replace(/\s+/g, "").toUpperCase();
  const ibanTitular = texto(fd, "ibanTitular");

  if (!multicaixaNumero && !iban) return falha("Indica pelo menos o Multicaixa Express ou o IBAN.");
  if (multicaixaNumero) {
    if (!multicaixaValido(multicaixaNumero)) return falha("O número do Multicaixa Express deve ter 9 dígitos (ex.: 923 456 789).");
    if (!multicaixaTitular) return falha("Indica o nome do titular do Multicaixa Express.");
  }
  if (iban) {
    if (!ibanValido(iban)) return falha("IBAN inválido. Confirma: começa por AO e tem 25 caracteres.");
    if (!ibanTitular) return falha("Indica o nome do titular da conta do IBAN.");
  }

  const pagamento: DadosPagamento = {
    multicaixaNumero,
    multicaixaTitular: multicaixaNumero ? multicaixaTitular : "",
    iban,
    ibanTitular: iban ? ibanTitular : "",
  };
  await db().collection("instituicoes").doc(inst.id).update({ pagamento });
  revalidatePath(rota);
  revalidatePath("/");
  return sucesso("Dados de pagamento guardados. Os próximos lembretes já os incluem.");
}

// ---------- Professores ----------

export async function criarProfessor(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const nome = texto(fd, "nome");
  const telefone = normalizarTelefone(texto(fd, "telefone"));
  if (!nome) return falha("Indica o nome do participante.");
  if (telefone.length < 11 || telefone.length > 15) return falha("Número de WhatsApp inválido.");
  const existe = await db()
    .collection("professores")
    .where("instituicaoId", "==", inst.id)
    .where("telefone", "==", telefone)
    .limit(1)
    .get();
  if (!existe.empty) return falha("Este número já está registado.");
  await db()
    .collection("professores")
    .add({ instituicaoId: inst.id, nome, telefone, ativo: true, criadoEm: new Date(), ultimoContactoEm: null });
  revalidatePath("/participantes");
  return sucesso(`${nome} adicionado.`);
}

/** Uma linha por professor: "Nome; 923456789" */
export async function importarProfessores(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const linhas = texto(fd, "lista").split(/\r?\n/);
  const dados = new Map<string, string>(); // telefone -> nome (sem repetidos na própria lista)
  for (const l of linhas) {
    const partes = l.split(/[;,\t]/).map((p) => p.trim()).filter(Boolean);
    if (partes.length < 2) continue;
    const telefone = normalizarTelefone(partes[partes.length - 1]);
    const nome = partes.slice(0, -1).join(" ");
    if (nome && telefone.length >= 11 && telefone.length <= 15 && !dados.has(telefone)) dados.set(telefone, nome);
  }
  if (dados.size === 0) return falha("Nenhuma linha válida. Usa o formato: Nome; 923456789");

  const existentes = new Set(
    (await listar<Professor>(db().collection("professores").where("instituicaoId", "==", inst.id))).map((p) => p.telefone),
  );
  const novos = [...dados].filter(([tel]) => !existentes.has(tel));
  for (let i = 0; i < novos.length; i += 400) {
    const lote = db().batch();
    for (const [telefone, nome] of novos.slice(i, i + 400)) {
      lote.set(db().collection("professores").doc(), {
        instituicaoId: inst.id,
        nome,
        telefone,
        ativo: true,
        criadoEm: new Date(),
        ultimoContactoEm: null,
      });
    }
    await lote.commit();
  }
  revalidatePath("/participantes");
  return sucesso(`${novos.length} participantes importados.`);
}

export async function alternarProfessor(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const ref = db().collection("professores").doc(texto(fd, "id"));
  const p = ler<Professor>(await ref.get());
  if (!p || p.instituicaoId !== inst.id) return falha("Participante não encontrado.");
  await ref.update({ ativo: !p.ativo });
  revalidatePath("/participantes");
  return sucesso(p.ativo ? `${p.nome} pausado: deixa de receber lembretes.` : `${p.nome} reativado.`);
}

export async function editarProfessor(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const id = texto(fd, "id");
  const rota = `/participantes/${id}`;
  const ref = db().collection("professores").doc(id);
  const p = ler<Professor>(await ref.get());
  if (!p || p.instituicaoId !== inst.id) return falha("Participante não encontrado.");

  const nome = texto(fd, "nome");
  const telefone = normalizarTelefone(texto(fd, "telefone"));
  if (!nome) return falha("Indica o nome do participante.");
  if (telefone.length < 11 || telefone.length > 15) return falha("Número de WhatsApp inválido.");
  const outro = await db()
    .collection("professores")
    .where("instituicaoId", "==", inst.id)
    .where("telefone", "==", telefone)
    .get();
  if (outro.docs.some((d) => d.id !== id)) return falha("Este número já pertence a outro participante.");

  await ref.update({ nome, telefone });
  revalidatePath("/participantes");
  revalidatePath(rota);
  return sucesso("Participante atualizado.");
}

/** Apaga o professor e o que depende dele: inscrições, comprovativos, pagamentos em mãos e mensagens. */
export async function apagarProfessor(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const r = await apagarProfessorCompleto(inst.id, texto(fd, "id"));
  revalidatePath("/participantes");
  revalidatePath("/");
  if (!r.ok) return falha(r.erro);
  return sucesso("Participante apagado, com os seus pagamentos e mensagens.", destino(fd));
}

// ---------- Campanhas ----------

async function professoresAtivosIds(instituicaoId: string): Promise<string[]> {
  const r = await db()
    .collection("professores")
    .where("instituicaoId", "==", instituicaoId)
    .where("ativo", "==", true)
    .get();
  return r.docs.map((d) => d.id);
}

/** Dos ids recebidos do formulário, só os professores ativos desta instituição. */
async function idsValidos(instituicaoId: string, ids: string[]): Promise<string[]> {
  const permitidos = new Set(await professoresAtivosIds(instituicaoId));
  return [...new Set(ids)].filter((id) => permitidos.has(id));
}

/** Campanha só se for da instituição da sessão. */
async function campanhaDaInstituicao(instituicaoId: string, id: string): Promise<Campanha | null> {
  const c = ler<Campanha>(await db().collection("campanhas").doc(id).get());
  return c && c.instituicaoId === instituicaoId ? c : null;
}

export async function criarCampanha(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const tipo = ["MENSAL", "SEMANAL"].includes(texto(fd, "repete")) ? (texto(fd, "repete") as "MENSAL" | "SEMANAL") : "UNICA";
  const recorrente = tipo !== "UNICA";
  const ano = inteiro(fd, "ano") || anoAtual();
  const nome = texto(fd, "nome") || `Campanha ${ano}`;
  const valorTotal = inteiro(fd, "valorTotal");
  const frequenciaHoras = Math.max(1, inteiro(fd, "frequenciaDias") * 24 || 48);

  if (!Number.isInteger(ano) || ano < 2000 || ano > 2100) return falha("Indica um ano válido.");
  if (!valorTotal || valorTotal <= 0) {
    return falha(tipo === "MENSAL" ? "Indica o valor da cota mensal em Kz." : tipo === "SEMANAL" ? "Indica o valor da cota semanal em Kz." : "Indica o valor total em Kz.");
  }

  // Calendário: pagamento único tem um prazo; as recorrentes têm um prazo por mês ou por semana, durante o ano todo.
  let prazoUnico: Date | null = null;
  let calendario: { primeiroPrazo: Date; periodos: number } | null = null;
  let diaMes: number | undefined;
  if (!recorrente) {
    const prazo = texto(fd, "prazo");
    if (!/^\d{4}-\d{2}-\d{2}$/.test(prazo)) return falha("Indica a data do prazo final.");
    prazoUnico = fimDoDia(prazo);
  } else {
    diaMes = tipo === "MENSAL" ? inteiro(fd, "diaMes") : undefined;
    if (tipo === "MENSAL" && (!Number.isInteger(diaMes) || (diaMes as number) < 1 || (diaMes as number) > 31)) {
      return falha("Indica o dia do mês do prazo (de 1 a 31).");
    }
    const diaSemana = Number(texto(fd, "diaSemana"));
    if (tipo === "SEMANAL" && !(Number.isInteger(diaSemana) && diaSemana >= 0 && diaSemana <= 6)) {
      return falha("Escolhe o dia da semana do prazo.");
    }
    const inicio = texto(fd, "inicio");
    if (inicio && !/^\d{4}-\d{2}-\d{2}$/.test(inicio)) return falha("A data de início não é válida.");
    calendario = calendarioDoAno({ periodicidade: tipo as "MENSAL" | "SEMANAL", ano, inicio: inicio || undefined, diaSemana, diaMes });
    if (!calendario) return falha(`Com essa data de início já não há nenhum prazo em ${ano}. Muda a data de início ou o ano.`);
    if (calendario.periodos > MAX_PERIODOS) return falha(`No máximo ${MAX_PERIODOS} períodos por campanha.`);
  }

  // Campanha de um ano anterior de onde copiar os participantes (e, se pedido, transitar as dívidas).
  const copiarDe = texto(fd, "copiarDe");
  const transitar = Boolean(fd.get("transitar"));
  let origem: Campanha | null = null;
  let origemInscs: Inscricao[] = [];
  if (copiarDe) {
    origem = await campanhaDaInstituicao(inst.id, copiarDe);
    if (!origem) return falha("A campanha a copiar não foi encontrada.");
    if (origem.recorrente) origem = (await atualizarPeriodos(origem)).campanha;
    if (transitar && new Date() <= origem.prazo) {
      return falha(
        `A campanha “${origem.nome}” ainda não terminou (último prazo: ${fmtData(origem.prazo)}). ` +
          "Cria a nova depois dessa data, ou desliga “Passar as dívidas para a nova campanha”.",
      );
    }
    origemInscs = await listar<Inscricao>(db().collection("inscricoes").where("campanhaId", "==", origem.id));
  }

  // Participantes: os da campanha copiada, ou os escolhidos no formulário (só ativos desta instituição).
  const modo = texto(fd, "modo");
  let ids: string[];
  if (origem) {
    ids = await idsValidos(inst.id, origemInscs.map((i) => i.professorId));
  } else if (modo === "escolher") {
    ids = await idsValidos(inst.id, fd.getAll("professorIds").map(String));
    if (ids.length === 0) return falha("Escolhe pelo menos um participante, ou muda para “Todos”.");
  } else if (modo === "nenhum") {
    ids = [];
  } else {
    ids = await professoresAtivosIds(inst.id);
  }

  const calendarioCampanha = calendario
    ? { recorrente: true, periodicidade: tipo as "MENSAL" | "SEMANAL", ...(diaMes ? { diaMes } : {}), periodos: calendario.periodos, primeiroPrazo: calendario.primeiroPrazo }
    : null;
  const ref = db().collection("campanhas").doc();
  await ref.set({
    instituicaoId: inst.id,
    nome,
    valorTotal,
    ano,
    prazo: calendarioCampanha ? prazoFinal(calendarioCampanha) : (prazoUnico as Date),
    frequenciaHoras,
    ativa: true,
    criadaEm: new Date(),
    recorrente,
    ...(calendarioCampanha ? { ...calendarioCampanha, periodosAplicados: periodosEmCurso(calendarioCampanha) } : {}),
  });

  // Dívida que transita e cotas próprias (só de quem continua a ser participante ativo)
  const saldos = new Map<string, number>();
  const cotas = new Map<string, number>();
  const idsSet = new Set(ids);
  for (const i of origemInscs) {
    if (!idsSet.has(i.professorId)) continue;
    if (i.valorPeriodo) cotas.set(i.professorId, i.valorPeriodo);
    const falta = Math.max(0, i.valorDevido - i.valorPago);
    if (transitar && falta > 0 && !i.valorTransitado) saldos.set(i.professorId, falta);
  }
  const base = { id: ref.id, instituicaoId: inst.id, valorTotal, ...(calendarioCampanha ?? {}) };
  const incluidos = await inscreverProfessores(base, ids, { saldos, cotas });

  // A dívida passa para a campanha nova: na antiga fica registada como "transitada" e a campanha fecha.
  let transitados = 0;
  if (origem && transitar) {
    const agora = new Date();
    for (const i of origemInscs) {
      const saldo = saldos.get(i.professorId);
      if (!saldo) continue;
      await db().collection("inscricoes").doc(i.id).update({
        valorTransitado: saldo,
        valorDevidoOriginal: i.valorDevido,
        valorDevido: i.valorPago,
        status: "QUITADO",
        quitadoEm: i.quitadoEm ?? agora,
      });
      transitados++;
    }
    await db().collection("campanhas").doc(origem.id).update({ ativa: false });
  }

  revalidatePath("/campanhas");
  revalidatePath("/");
  const aviso =
    transitados > 0 ? ` A dívida de ${transitados} ${transitados === 1 ? "participante" : "participantes"} passou de “${origem?.nome}” para esta campanha.` : "";
  return sucesso(
    (incluidos > 0 ? `Campanha criada com ${incluidos} ${incluidos === 1 ? "participante" : "participantes"}.` : "Campanha criada.") + aviso,
    `/campanhas/${ref.id}`,
  );
}

export async function inscreverTodos(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const id = texto(fd, "id");
  const c = await campanhaDaInstituicao(inst.id, id);
  if (!c) return falha("Campanha não encontrada.");
  const n = await inscreverProfessores(c, await professoresAtivosIds(inst.id), { desdeAgora: true });
  revalidatePath(`/campanhas/${id}`);
  return sucesso(`${n} participantes incluídos.`);
}

/** Adiciona à campanha só os professores escolhidos (os que já estão são ignorados). */
export async function adicionarProfessoresACampanha(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const id = texto(fd, "id");
  const c = await campanhaDaInstituicao(inst.id, id);
  if (!c) return falha("Campanha não encontrada.");
  const ids = await idsValidos(inst.id, fd.getAll("professorIds").map(String));
  if (ids.length === 0) return falha("Escolhe pelo menos um participante.");
  const n = await inscreverProfessores(c, ids, { desdeAgora: true });
  revalidatePath(`/campanhas/${id}`);
  revalidatePath("/");
  return sucesso(n > 0 ? `${n} ${n === 1 ? "participante adicionado" : "participantes adicionados"} à campanha.` : "Esses participantes já estavam na campanha.");
}

/** Ações sobre os professores marcados na tabela de uma campanha: enviar lembrete ou remover. */
export async function acaoEmLote(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const campanhaId = texto(fd, "campanhaId");
  const ids = [...new Set(fd.getAll("inscricaoIds").map(String))];
  const acao = texto(fd, "acao");
  if (!(await campanhaDaInstituicao(inst.id, campanhaId))) return falha("Campanha não encontrada.");
  if (ids.length === 0) return falha("Marca pelo menos um participante.");
  if (ids.length > 100) return falha("Marca no máximo 100 participantes de cada vez.");

  if (acao === "remover") {
    const n = await removerInscricoesCompleto(inst.id, campanhaId, ids);
    revalidatePath(`/campanhas/${campanhaId}`);
    revalidatePath("/");
    return sucesso(`${n} ${n === 1 ? "participante removido" : "participantes removidos"} da campanha.`);
  }
  if (acao === "lembrar") {
    const r = await enviarLembretesSelecionados(inst.id, campanhaId, ids);
    revalidatePath(`/campanhas/${campanhaId}`);
    if (r.motivo) return falha(r.motivo);
    const extra = r.ignorados > 0 ? ` ${r.ignorados} ignorados (já pagaram ou estão pausados).` : "";
    if (r.falhas > 0) return falha(`Enviados: ${r.enviados}. Falharam: ${r.falhas}.${extra}`);
    return sucesso(r.enviados > 0 ? `Lembrete enviado a ${r.enviados} ${r.enviados === 1 ? "participante" : "participantes"}.${extra}` : `Nada enviado.${extra}`);
  }
  return falha("Escolhe uma ação.");
}

export async function editarCampanha(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const id = texto(fd, "id");
  const rota = `/campanhas/${id}`;
  const c = await campanhaDaInstituicao(inst.id, id);
  if (!c) return falha("Campanha não encontrada.");

  const nome = texto(fd, "nome");
  const valorTotal = inteiro(fd, "valorTotal");
  const prazo = texto(fd, "prazo");
  const ano = inteiro(fd, "ano") || anoDaCampanha(c);
  const frequenciaHoras = Math.max(1, inteiro(fd, "frequenciaDias") * 24 || 48);
  if (!nome) return falha("Indica o nome da campanha.");
  if (!valorTotal || valorTotal <= 0) return falha("Indica o valor em Kz.");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(prazo)) return falha("Indica a data do prazo.");
  if (!Number.isInteger(ano) || ano < 2000 || ano > 2100) return falha("Indica um ano válido.");

  if (c.recorrente) {
    // Mensal ou semanal: o valor é a cota. Mudar a cota, o 1.º prazo ou o nº de períodos recalcula o devido de todos.
    const periodos = inteiro(fd, "periodos");
    if (!Number.isInteger(periodos) || periodos < 1 || periodos > MAX_PERIODOS) {
      return falha(`Indica quantos períodos dura a campanha (de 1 a ${MAX_PERIODOS}).`);
    }
    const mensal = periodicidadeDe(c) === "MENSAL";
    const primeiroPrazo = fimDoDia(prazo);
    const diaMes = mensal ? Number(prazo.slice(8, 10)) : undefined;
    const calendario = { ...c, periodos, primeiroPrazo, ...(diaMes ? { diaMes } : {}) };
    const novos = {
      nome,
      valorTotal,
      ano,
      periodos,
      primeiroPrazo,
      ...(diaMes ? { diaMes } : {}),
      prazo: prazoFinal(calendario),
      frequenciaHoras,
    };
    await db().collection("campanhas").doc(id).update(novos);
    await atualizarPeriodos({ ...c, ...novos }, { forcar: true });
    revalidatePath("/campanhas");
    revalidatePath(rota);
    revalidatePath("/");
    return sucesso("Campanha atualizada. Os valores devidos foram recalculados.");
  }

  await db().collection("campanhas").doc(id).update({ nome, valorTotal, ano, prazo: fimDoDia(prazo), frequenciaHoras });

  let atualizados = 0;
  if (fd.get("aplicarATodos") && valorTotal !== c.valorTotal) {
    const inscs = await listar<Inscricao>(db().collection("inscricoes").where("campanhaId", "==", id));
    for (const i of inscs) {
      if (i.status === "QUITADO" || i.valorDevido === valorTotal) continue;
      const r = await alterarValorDevido(inst.id, i.id, valorTotal);
      if (r.ok) atualizados++;
    }
  }
  revalidatePath("/campanhas");
  revalidatePath(rota);
  revalidatePath("/");
  return sucesso(atualizados > 0 ? `Campanha atualizada. Valor devido alterado a ${atualizados} participantes.` : "Campanha atualizada.");
}

/** Apaga a campanha com as inscrições, comprovativos e pagamentos em mãos dela. */
export async function apagarCampanha(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const r = await apagarCampanhaCompleta(inst.id, texto(fd, "id"));
  revalidatePath("/campanhas");
  revalidatePath("/");
  if (!r.ok) return falha(r.erro);
  return sucesso("Campanha apagada.", destino(fd));
}

export async function removerInscricao(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const rota = `/campanhas/${texto(fd, "campanhaId")}`;
  const r = await removerInscricaoCompleta(inst.id, texto(fd, "inscricaoId"));
  revalidatePath(rota);
  revalidatePath("/");
  if (!r.ok) return falha(r.erro);
  return sucesso("Participante removido da campanha.");
}

export async function alternarCampanha(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const id = texto(fd, "id");
  const c = await campanhaDaInstituicao(inst.id, id);
  if (!c) return falha("Campanha não encontrada.");
  await db().collection("campanhas").doc(id).update({ ativa: !c.ativa });
  revalidatePath(`/campanhas/${id}`);
  revalidatePath("/campanhas");
  revalidatePath("/");
  return sucesso(c.ativa ? "Campanha encerrada." : "Campanha reativada.");
}

export async function editarValorDevido(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const campanhaId = texto(fd, "campanhaId");
  const rota = `/campanhas/${campanhaId}`;
  const r = await alterarValorDevido(inst.id, texto(fd, "inscricaoId"), inteiro(fd, "valor"));
  revalidatePath(rota);
  revalidatePath("/");
  if (!r.ok) return falha(r.erro);
  return sucesso("Valor devido atualizado.");
}

/** Pagamento em mãos: o admin regista o valor recebido. */
export async function registarPagamentoMao(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const campanhaId = texto(fd, "campanhaId");
  const rota = `/campanhas/${campanhaId}`;
  const r = await registarPagamentoManual(
    inst.id,
    texto(fd, "inscricaoId"),
    inteiro(fd, "valor"),
    texto(fd, "nota"),
    Boolean(fd.get("avisar")),
  );
  revalidatePath(rota);
  revalidatePath("/");
  if (!r.ok) return falha(r.erro);
  return sucesso("Pagamento em mãos registado.");
}

export async function anularPagamentoMao(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const campanhaId = texto(fd, "campanhaId");
  const rota = `/campanhas/${campanhaId}`;
  const r = await anularPagamentoManual(inst.id, texto(fd, "id"));
  revalidatePath(rota);
  revalidatePath("/");
  if (!r.ok) return falha(r.erro);
  return sucesso("Pagamento anulado e total recalculado.");
}

export async function enviarLembretesAgora(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const id = texto(fd, "id");
  if (!(await campanhaDaInstituicao(inst.id, id))) return falha("Campanha não encontrada.");
  const r = await executarLembretes({ forcar: true, campanhaId: id, instituicaoId: inst.id });
  revalidatePath(`/campanhas/${id}`);
  if (r.falhas) return falha(`Lembretes enviados: ${r.enviados}. Falhas: ${r.falhas}.`);
  return sucesso(`Lembretes enviados: ${r.enviados}.`);
}

// ---------- Comprovativos ----------

export async function aprovarAction(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const r = await aprovarComprovativo(inst.id, texto(fd, "id"), inteiro(fd, "valor"));
  revalidatePath("/comprovativos");
  revalidatePath("/");
  if (!r.ok) return falha(r.erro);
  return sucesso("Comprovativo aprovado. O participante foi avisado no WhatsApp.");
}

export async function rejeitarAction(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const r = await rejeitarComprovativo(inst.id, texto(fd, "id"), texto(fd, "motivo"));
  revalidatePath("/comprovativos");
  revalidatePath("/");
  if (!r.ok) return falha(r.erro);
  return sucesso("Comprovativo rejeitado. O participante foi avisado no WhatsApp.");
}

export async function corrigirComprovativoAction(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const estado = texto(fd, "estado");
  const r = await corrigirValorAprovado(inst.id, texto(fd, "id"), inteiro(fd, "valor"));
  revalidatePath("/comprovativos");
  revalidatePath("/");
  if (!r.ok) return falha(r.erro);
  return sucesso("Valor corrigido e total recalculado.");
}

export async function apagarComprovativoAction(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  const estado = texto(fd, "estado");
  const r = await apagarComprovativo(inst.id, texto(fd, "id"));
  revalidatePath("/comprovativos");
  revalidatePath("/");
  if (!r.ok) return falha(r.erro);
  return sucesso("Comprovativo apagado.");
}

// ---------- SMS ----------

const MAX_DESTINATARIOS_SMS = 500;

/** SMS escrito pelo admin: o mesmo texto para todos os professores ativos ou só para os escolhidos. */
export async function enviarSmsAction(fd: FormData): Promise<Resultado> {
  const inst = await requireInstituicaoAcao();
  if (!smsAtivo(inst)) return falha("O SMS não está ativo para esta instituição. Pede ao dono da plataforma para o ativar.");
  if (!smsConfigurado()) return falha("O SMS ainda não está configurado no servidor (faltam OMBALA_TOKEN e OMBALA_FROM).");

  const corpo = texto(fd, "mensagem");
  if (!corpo) return falha("Escreve a mensagem.");
  const final = paraSms(fd.get("assinar") ? `${inst.nome}: ${corpo}` : corpo);
  if (!final) return falha("A mensagem ficou vazia.");
  if (septetos(final) > MAX_SEPTETOS) {
    return falha(`Mensagem demasiado longa: o máximo são ${MAX_SEPTETOS} caracteres (${MAX_SEGMENTOS} SMS).`);
  }

  const ativos = await listar<Professor>(
    db().collection("professores").where("instituicaoId", "==", inst.id).where("ativo", "==", true),
  );
  let alvos = ativos;
  if (texto(fd, "modo") === "escolher") {
    const escolhidos = new Set(fd.getAll("professorIds").map(String));
    alvos = ativos.filter((p) => escolhidos.has(p.id));
    if (alvos.length === 0) return falha("Escolhe pelo menos um participante, ou muda para “Todos”.");
  }
  if (alvos.length === 0) return falha("Ainda não há participantes ativos.");
  if (alvos.length > MAX_DESTINATARIOS_SMS) {
    return falha(`No máximo ${MAX_DESTINATARIOS_SMS} participantes de cada vez. Escolhe menos.`);
  }

  const r = await enviarSmsAosProfessores(inst, alvos, final);
  revalidatePath("/sms");
  revalidatePath("/");
  const detalhe = r.erro ? ` (${r.erro})` : "";
  if (r.enviados === 0) return falha(`Nenhum SMS foi enviado${detalhe}.`);
  if (r.falhas > 0) return falha(`SMS enviado a ${r.enviados}. Falharam ${r.falhas}${detalhe}.`);
  return sucesso(`SMS enviado a ${r.enviados} ${r.enviados === 1 ? "participante" : "participantes"}.`);
}
