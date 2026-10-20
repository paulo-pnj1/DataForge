"use client";

import { useRouter } from "next/navigation";
import { useTransition, type CSSProperties, type FormEvent, type ReactNode } from "react";
import type { Resultado } from "@/lib/resultado";
import { avisar } from "./Toaster";

/**
 * Formulário que corre uma ação do servidor SEM recarregar a página:
 * mostra uma mensagem rápida, atualiza só os dados que mudaram e deixa o botão em "a enviar" até acabar.
 * `reiniciar` limpa os campos depois de sucesso (formulários de criar).
 */
export function FormAcao({
  action,
  children,
  className,
  style,
  id,
  reiniciar,
}: {
  action: (dados: FormData) => Promise<Resultado>;
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  id?: string;
  reiniciar?: boolean;
}) {
  const router = useRouter();
  const [pendente, iniciar] = useTransition();

  function aoSubmeter(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (pendente) return;
    const form = e.currentTarget;
    const botao = (e.nativeEvent as SubmitEvent).submitter as HTMLButtonElement | null;
    const dados = new FormData(form, botao);
    iniciar(async () => {
      try {
        const r = await action(dados);
        avisar(r.mensagem, r.ok ? "ok" : "erro");
        if (r.ok) {
          if (reiniciar) form.reset();
          if (r.ir) router.push(r.ir);
        }
      } catch {
        avisar("Não foi possível concluir. Tenta outra vez.", "erro");
      }
    });
  }

  return (
    <form id={id} className={className} style={style} onSubmit={aoSubmeter} aria-busy={pendente} data-pendente={pendente ? "" : undefined}>
      {children}
    </form>
  );
}
