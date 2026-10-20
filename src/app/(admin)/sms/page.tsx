import { FormAcao } from "@/components/FormAcao";
import { FormularioSms } from "@/components/FormularioSms";
import { Tag } from "@/components/ui";
import { requireInstituicao } from "@/lib/auth";
import { CANAIS, canalDe, smsAtivo } from "@/lib/canais";
import { ultimasMensagens } from "@/lib/consultas";
import { db, listar } from "@/lib/db";
import { fmtDataHora } from "@/lib/format";
import { smsConfigurado } from "@/lib/sms";
import type { Professor } from "@/lib/tipos";
import { enviarSmsAction } from "../../actions";

export default async function Sms() {
  const inst = await requireInstituicao();
  const ativo = smsAtivo(inst);
  const configurado = smsConfigurado();

  const [professores, recentes] = ativo
    ? await Promise.all([
        listar<Professor>(
          db().collection("professores").where("instituicaoId", "==", inst.id).where("ativo", "==", true),
        ),
        ultimasMensagens(inst.id, 60),
      ])
    : [[], []];
  professores.sort((a, b) => a.nome.localeCompare(b.nome, "pt"));
  const enviados = recentes.filter((m) => m.canal === "SMS").slice(0, 10);
  const rotuloCanal = CANAIS.find((c) => c.valor === canalDe(inst))?.rotulo;

  return (
    <>
      <div className="topo">
        <div>
          <h1>SMS</h1>
          <p>Envia um aviso por SMS aos participantes. Chega a qualquer telemóvel, mesmo sem internet nem WhatsApp.</p>
        </div>
      </div>

      {!ativo && (
        <div className="aviso erro" role="alert">
          O SMS não está ativo para esta instituição. Pede ao dono da plataforma para o ativar.
        </div>
      )}
      {ativo && !configurado && (
        <div className="aviso erro" role="alert">
          O SMS ainda não está configurado no servidor (faltam OMBALA_TOKEN e OMBALA_FROM).
        </div>
      )}

      {ativo && (
        <>
          <section className="bloco">
            <h2>Novo SMS</h2>
            <FormAcao reiniciar action={enviarSmsAction}>
              <FormularioSms
                nomeInstituicao={inst.nome}
                professores={professores.map((p) => ({ id: p.id, nome: p.nome, telefone: p.telefone }))}
                desativado={!configurado}
              />
            </FormAcao>
            <p className="suave" style={{ fontSize: 13, marginTop: 12 }}>
              Mensagens automáticas desta instituição: {rotuloCanal}. SMS só para telemóveis angolanos (9 dígitos).
            </p>
          </section>

          <section className="bloco">
            <h2>Últimos SMS</h2>
            {enviados.length === 0 ? (
              <p className="vazio">Ainda não foi enviado nenhum SMS.</p>
            ) : (
              <div className="tabela-rolagem">
                <table className="tabela">
                  <thead>
                    <tr><th>Quando</th><th>Participante</th><th>Tipo</th><th>Estado</th></tr>
                  </thead>
                  <tbody>
                    {enviados.map((m) => (
                      <tr key={m.id}>
                        <td data-label="Quando" className="suave">{fmtDataHora(m.criadaEm)}</td>
                        <td data-label="Participante">{m.professor?.nome ?? "—"}</td>
                        <td data-label="Tipo">{m.tipo}</td>
                        <td data-label="Estado">
                          {m.status === "failed" ? (
                            <span title={m.erro ?? ""}><Tag tom="erro">falhou</Tag> <span className="suave">{m.erro}</span></span>
                          ) : (
                            <Tag tom="aviso">enviado</Tag>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </>
  );
}
