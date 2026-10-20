"use client";

import type { ReactNode } from "react";

/** Botão de submit que pede confirmação antes de enviar o formulário (apagar, remover, anular). */
export function BotaoConfirmar({
  mensagem,
  className = "btn perigo mini",
  name,
  value,
  disabled,
  children,
}: {
  mensagem: string;
  className?: string;
  name?: string;
  value?: string;
  disabled?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="submit"
      className={className}
      name={name}
      value={value}
      disabled={disabled}
      onClick={(e) => {
        if (!window.confirm(mensagem)) e.preventDefault();
      }}
    >
      {children}
    </button>
  );
}
