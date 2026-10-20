import { redirect } from "next/navigation";
import { contarPendentes } from "@/lib/consultas";
import { requireSessao } from "@/lib/auth";
import { obterInstituicao } from "@/lib/instituicoes";
import { smsAtivo } from "@/lib/canais";
import { Shell } from "@/components/Shell";
import { logoutAction, sairInstituicao } from "../actions";

export default async function AdminLayout({ children }: { children: React.ReactNode }) {
  const sessao = await requireSessao();
  const inst = sessao.instituicaoId ? await obterInstituicao(sessao.instituicaoId) : null;
  if (sessao.papel === "inst" && (!inst || !inst.ativa)) redirect("/login");
  const pendentes = inst ? await contarPendentes(inst.id) : 0;

  const itens = [
    ...(sessao.papel === "super" ? [{ href: "/instituicoes", rotulo: "Instituições" }] : []),
    ...(inst
      ? [
          { href: "/", rotulo: "Resumo" },
          { href: "/comprovativos", rotulo: "Comprovativos", contagem: pendentes },
          { href: "/campanhas", rotulo: "Campanhas" },
          { href: "/participantes", rotulo: "Participantes" },
          ...(smsAtivo(inst) ? [{ href: "/sms", rotulo: "SMS" }] : []),
          { href: "/pagamento", rotulo: "Dados de pagamento" },
        ]
      : []),
  ];

  const rodape = (
    <>
      {sessao.papel === "super" && inst && (
        <form action={sairInstituicao}>
          <button className="sair" type="submit">Sair de {inst.nome}</button>
        </form>
      )}
      <form action={logoutAction}>
        <button className="sair" type="submit">Terminar sessão</button>
      </form>
    </>
  );

  return (
    <Shell itens={itens} subtitulo={inst ? inst.nome : "muanaOS"} rodape={rodape}>
      {children}
    </Shell>
  );
}
