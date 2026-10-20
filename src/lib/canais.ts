import type { CanalEnvio, Instituicao } from "./tipos";

/** Como cada instituição envia as mensagens automáticas (lembretes, aprovado, rejeitado, parabéns). */
export const CANAIS: { valor: CanalEnvio; rotulo: string }[] = [
  { valor: "WHATSAPP", rotulo: "Só WhatsApp (SMS desligado)" },
  { valor: "WHATSAPP_COM_SMS_RESERVA", rotulo: "WhatsApp, e SMS se o WhatsApp falhar" },
  { valor: "SMS", rotulo: "Só SMS" },
  { valor: "WHATSAPP_E_SMS", rotulo: "WhatsApp e SMS ao mesmo tempo" },
];

export const canalDe = (i: Pick<Instituicao, "canal">): CanalEnvio => i.canal ?? "WHATSAPP";

export const canalValido = (v: string): CanalEnvio => CANAIS.find((c) => c.valor === v)?.valor ?? "WHATSAPP";

/** O SMS está ligado para esta instituição (página SMS e envios automáticos por SMS). */
export const smsAtivo = (i: Pick<Instituicao, "canal">): boolean => canalDe(i) !== "WHATSAPP";
