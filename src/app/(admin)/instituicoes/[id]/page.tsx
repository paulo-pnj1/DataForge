import { FormAcao } from "@/components/FormAcao";
import Link from "next/link";
import { notFound } from "next/navigation";
import { db, ler } from "@/lib/db";
import type { Instituicao } from "@/lib/tipos";
import { CANAIS, canalDe } from "@/lib/canais";
import { requireSuper } from "@/lib/auth";
import { Aviso } from "@/components/ui";
import { BotaoConfirmar } from "@/components/BotaoConfirmar";
import { apagarInstituicao, editarInstituicao, redefinirSenhaInstituicao } from "../../../actions";

export default async function EditarInstituicao({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ erro?: string; ok?: string }>;
}) {
  await requireSuper();
  const { id } = await params;
  const { erro, ok } = await searchParams;
  const i = ler<Instituicao>(await db().collection("instituicoes").doc(id).get());
  if (!i) notFound();

  return (
    <>
      <div className="topo">
        <div>
          <Link className="voltar" href="/instituicoes">← Instituições</Link>
          <h1>{i.nome}</h1>
          <p>{i.adminEmail} · {i.ativa ? "ativa" : "desativada"}</p>
        </div>
      </div>
      <Aviso erro={erro} ok={ok} />

      <section className="bloco">
        <h2>Editar dados</h2>
        <FormAcao action={editarInstituicao} style={{ display: "grid", gap: 16 }}>
          <input type="hidden" name="id" value={i.id} />
          <div className="campos">
            <div>
              <label htmlFor="nome">Nome (aparece nas mensagens)</label>
              <input id="nome" name="nome" defaultValue={i.nome} required />
            </div>
            <div>
              <label htmlFor="adminEmail">Email do admin</label>
              <input id="adminEmail" name="adminEmail" type="email" defaultValue={i.adminEmail} required />
            </div>
          </div>
          <div className="campos">
            <div>
              <label htmlFor="canal">Canal das mensagens automáticas</label>
              <select id="canal" name="canal" defaultValue={canalDe(i)}>
                {CANAIS.map((c) => <option key={c.valor} value={c.valor}>{c.rotulo}</option>)}
              </select>
            </div>
          </div>
          <p className="suave" style={{ fontSize: 13 }}>
            Cada SMS gasta créditos da tua conta Ombala. Com qualquer opção que inclua SMS, a instituição passa a ter a página SMS no menu.
          </p>
          <div className="linha-check">
            <input id="tomCristao" name="tomCristao" type="checkbox" defaultChecked={i.tomCristao} />
            <label htmlFor="tomCristao">Frases cristãs nas mensagens</label>
          </div>
          <div><button className="btn" type="submit">Guardar alterações</button></div>
        </FormAcao>
      </section>

      <section className="bloco">
        <h2>Nova senha do admin</h2>
        <FormAcao reiniciar action={redefinirSenhaInstituicao} className="campos">
          <input type="hidden" name="id" value={i.id} />
          <div>
            <label htmlFor="senha">Senha (mín. 8 caracteres)</label>
            <input id="senha" name="senha" type="password" minLength={8} autoComplete="new-password" required />
          </div>
          <div><button className="btn sec" type="submit">Guardar senha</button></div>
        </FormAcao>
      </section>

      <section className="bloco zona-perigo">
        <h2>Apagar instituição</h2>
        <p className="suave">
          Apaga {i.nome} e <b>tudo</b> o que lhe pertence: participantes, campanhas, comprovativos (e ficheiros), pagamentos e mensagens.
          Não dá para desfazer. Se só queres suspender o acesso, usa “Desativar” na lista.
        </p>
        <FormAcao action={apagarInstituicao} style={{ display: "grid", gap: 12, maxWidth: 420 }}>
          <input type="hidden" name="id" value={i.id} />
          <div>
            <label htmlFor="confirmacao">Escreve o nome exato para confirmar: <b>{i.nome}</b></label>
            <input id="confirmacao" name="confirmacao" autoComplete="off" required />
          </div>
          <div>
            <BotaoConfirmar className="btn perigo" mensagem={`Apagar ${i.nome} e todos os dados? Não dá para desfazer.`}>
              Apagar instituição
            </BotaoConfirmar>
          </div>
        </FormAcao>
      </section>
    </>
  );
}
