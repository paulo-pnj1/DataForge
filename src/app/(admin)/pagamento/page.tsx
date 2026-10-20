import { FormAcao } from "@/components/FormAcao";
import { Aviso } from "@/components/ui";
import { CopiarBotao } from "@/components/CopiarBotao";
import { requireInstituicao } from "@/lib/auth";
import { fmtIban, fmtNumero, pagamentoEmLinha, temDadosPagamento } from "@/lib/dadosPagamento";
import { guardarDadosPagamento } from "../../actions";

export default async function Pagamento({
  searchParams,
}: {
  searchParams: Promise<{ erro?: string; ok?: string }>;
}) {
  const inst = await requireInstituicao();
  const { erro, ok } = await searchParams;
  const p = inst.pagamento;
  const configurado = temDadosPagamento(inst);

  return (
    <>
      <div className="topo">
        <div>
          <h1>Dados de pagamento</h1>
          <p>Para onde os participantes fazem a transferência. Estes dados vão em cada lembrete de WhatsApp.</p>
        </div>
      </div>
      <Aviso erro={erro} ok={ok} />
      {!configurado && !erro && !ok && (
        <div className="aviso erro" role="alert">
          Ainda não há dados de pagamento. Os lembretes dizem aos participantes para os pedirem à instituição.
        </div>
      )}

      <FormAcao action={guardarDadosPagamento}>
        <section className="bloco">
          <h2>Multicaixa Express</h2>
          <div className="campos">
            <div>
              <label htmlFor="multicaixaNumero">Número associado</label>
              <input
                id="multicaixaNumero"
                name="multicaixaNumero"
                inputMode="tel"
                placeholder="923 456 789"
                defaultValue={p?.multicaixaNumero ? fmtNumero(p.multicaixaNumero) : ""}
              />
            </div>
            <div>
              <label htmlFor="multicaixaTitular">Nome do titular</label>
              <input id="multicaixaTitular" name="multicaixaTitular" defaultValue={p?.multicaixaTitular ?? ""} />
            </div>
          </div>
        </section>

        <section className="bloco">
          <h2>Transferência bancária (IBAN)</h2>
          <div className="campos">
            <div>
              <label htmlFor="iban">IBAN</label>
              <input
                id="iban"
                name="iban"
                placeholder="AO06 0000 0000 0000 0000 0000 0"
                autoCapitalize="characters"
                defaultValue={p?.iban ? fmtIban(p.iban) : ""}
              />
            </div>
            <div>
              <label htmlFor="ibanTitular">Nome do titular da conta</label>
              <input id="ibanTitular" name="ibanTitular" defaultValue={p?.ibanTitular ?? ""} />
            </div>
          </div>
        </section>

        <div className="acoes" style={{ marginBottom: 20 }}>
          <button className="btn" type="submit">Guardar dados</button>
        </div>
      </FormAcao>

      {configurado && p && (
        <section className="bloco">
          <h2>Copiar</h2>
          <div className="tabela-rolagem">
            <table className="tabela">
              <thead><tr><th>Método</th><th>Dados</th><th></th></tr></thead>
              <tbody>
                {p.multicaixaNumero && (
                  <tr>
                    <td data-label="Método"><strong>Multicaixa Express</strong><div className="suave" style={{ fontSize: 13 }}>Titular: {p.multicaixaTitular}</div></td>
                    <td data-label="Dados">{fmtNumero(p.multicaixaNumero)}</td>
                    <td className="celula-acoes num"><CopiarBotao valor={p.multicaixaNumero} rotulo="Copiar número" /></td>
                  </tr>
                )}
                {p.iban && (
                  <tr>
                    <td data-label="Método"><strong>IBAN</strong><div className="suave" style={{ fontSize: 13 }}>Titular: {p.ibanTitular}</div></td>
                    <td data-label="Dados">{fmtIban(p.iban)}</td>
                    <td className="celula-acoes num"><CopiarBotao valor={p.iban} rotulo="Copiar IBAN" /></td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {configurado && (
        <section className="bloco">
          <h2>Assim aparece no lembrete</h2>
          <p style={{ color: "var(--tinta-suave)" }}>&ldquo;… Paga por: {pagamentoEmLinha(inst)}. …&rdquo;</p>
        </section>
      )}
    </>
  );
}
