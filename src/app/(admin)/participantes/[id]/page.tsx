import { FormAcao } from "@/components/FormAcao";
import Link from "next/link";
import { notFound } from "next/navigation";
import { db, ler, listar, porIds } from "@/lib/db";
import type { Campanha, Inscricao, Professor } from "@/lib/tipos";
import { Aviso, Progresso, Tag, rotuloInscricao, tomInscricao } from "@/components/ui";
import { BotaoConfirmar } from "@/components/BotaoConfirmar";
import { requireInstituicao } from "@/lib/auth";
import { kz } from "@/lib/format";
import { atualizarVariasCampanhas } from "@/lib/recorrencia";
import { apagarProfessor, editarProfessor } from "../../../actions";

export default async function EditarProfessor({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ erro?: string; ok?: string }>;
}) {
  const inst = await requireInstituicao();
  const { id } = await params;
  const { erro, ok } = await searchParams;

  const p = ler<Professor>(await db().collection("professores").doc(id).get());
  if (!p || p.instituicaoId !== inst.id) notFound();

  let inscricoes = await listar<Inscricao>(db().collection("inscricoes").where("professorId", "==", id));
  const campanhas = await porIds<Campanha>("campanhas", inscricoes.map((i) => i.campanhaId));
  if (await atualizarVariasCampanhas([...campanhas.values()])) {
    inscricoes = await listar<Inscricao>(db().collection("inscricoes").where("professorId", "==", id)); // com o mês novo somado
  }

  return (
    <>
      <div className="topo">
        <div>
          <Link className="voltar" href="/participantes">← Participantes</Link>
          <h1>{p.nome}</h1>
          <p>+{p.telefone} · {p.ativo ? "recebe lembretes" : "pausado"}</p>
        </div>
      </div>
      <Aviso erro={erro} ok={ok} />

      <section className="bloco">
        <h2>Editar dados</h2>
        <FormAcao action={editarProfessor} style={{ display: "grid", gap: 16 }}>
          <input type="hidden" name="id" value={p.id} />
          <div className="campos">
            <div>
              <label htmlFor="nome">Nome</label>
              <input id="nome" name="nome" defaultValue={p.nome} required />
            </div>
            <div>
              <label htmlFor="telefone">WhatsApp</label>
              <input id="telefone" name="telefone" inputMode="tel" defaultValue={p.telefone} required />
            </div>
          </div>
          <div><button className="btn" type="submit">Guardar alterações</button></div>
        </FormAcao>
      </section>

      <section className="bloco">
        <h2>Campanhas</h2>
        {inscricoes.length === 0 ? (
          <p className="vazio">Este participante ainda não está em nenhuma campanha.</p>
        ) : (
          <div className="grelha">
            {inscricoes.map((i) => {
              const c = campanhas.get(i.campanhaId);
              if (!c) return null;
              return (
                <Link key={i.id} href={`/campanhas/${c.id}`} className="bloco campanha-cartao" style={{ marginBottom: 0 }}>
                  <h3>{c.nome}</h3>
                  <div style={{ margin: "6px 0 10px" }}>
                    <Tag tom={tomInscricao[i.status]}>{rotuloInscricao[i.status]}</Tag>
                  </div>
                  <Progresso pago={i.valorPago} total={i.valorDevido} />
                  <div className="meta"><span>Pago {kz(i.valorPago)}</span><span>de {kz(i.valorDevido)}</span></div>
                </Link>
              );
            })}
          </div>
        )}
      </section>

      <section className="bloco zona-perigo">
        <h2>Apagar participante</h2>
        <p className="suave">
          Apaga o participante e também as inscrições, comprovativos, pagamentos em mãos e mensagens dele. Não dá para desfazer.
          Se só queres parar os lembretes, usa “Pausar” na lista.
        </p>
        <FormAcao action={apagarProfessor}>
          <input type="hidden" name="ir" value="/participantes" />
          <input type="hidden" name="id" value={p.id} />
          <BotaoConfirmar className="btn perigo" mensagem={`Apagar ${p.nome} e todos os dados dele? Não dá para desfazer.`}>
            Apagar {p.nome}
          </BotaoConfirmar>
        </FormAcao>
      </section>
    </>
  );
}
