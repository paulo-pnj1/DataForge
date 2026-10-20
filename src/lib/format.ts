const TZ = "Africa/Luanda";

export function kz(n: number): string {
  return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".") + " Kz";
}

/** Só dígitos; números angolanos de 9 dígitos ganham o indicativo 244. */
export function normalizarTelefone(raw: string): string {
  let d = raw.replace(/\D/g, "");
  if (d.startsWith("00")) d = d.slice(2);
  if (d.length === 9 && d.startsWith("9")) d = "244" + d;
  return d;
}

export function primeiroNome(nome: string): string {
  return nome.trim().split(/\s+/)[0] ?? nome;
}

export function diasRestantes(prazo: Date, agora = new Date()): number {
  return Math.ceil((prazo.getTime() - agora.getTime()) / 86_400_000);
}

export function fmtData(d: Date): string {
  return d.toLocaleDateString("pt-PT", { timeZone: TZ, day: "2-digit", month: "2-digit", year: "numeric" });
}

export function fmtDataHora(d: Date): string {
  return d.toLocaleString("pt-PT", {
    timeZone: TZ,
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function horaEmLuanda(d = new Date()): number {
  return Number(
    new Intl.DateTimeFormat("en-GB", { hour: "2-digit", hourCycle: "h23", timeZone: TZ }).format(d),
  );
}

/** "2026-11-30" -> fim desse dia em Luanda (UTC+1) */
export function fimDoDia(data: string): Date {
  return new Date(`${data}T23:59:59+01:00`);
}

export function pluralDias(n: number): string {
  return `${n} ${n === 1 ? "dia" : "dias"}`;
}

/** Data em Luanda no formato "2026-11-30" (para <input type="date">). */
export function dataISO(d: Date): string {
  return d.toLocaleDateString("sv-SE", { timeZone: TZ });
}
