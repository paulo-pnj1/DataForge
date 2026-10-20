/** O que uma ação do painel devolve ao navegador: uma mensagem e, opcionalmente, para onde ir. */
export type Resultado = { ok: boolean; mensagem: string; ir?: string };

export const sucesso = (mensagem: string, ir?: string): Resultado => ({ ok: true, mensagem, ir });
export const falha = (mensagem: string): Resultado => ({ ok: false, mensagem });
