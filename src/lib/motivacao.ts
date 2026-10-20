import { kz } from "./format";

const FRASES_CRISTAS = [
  "Quem semeia com generosidade, colhe com generosidade.",
  "Passo a passo chegamos lá: cada contribuição conta.",
  "Quem paga cedo descansa mais cedo.",
  "A tua fidelidade faz a diferença. Obrigado!",
  "Um pequeno valor hoje evita a correria no último dia.",
  "Deus ama quem dá com alegria. Estás quase lá!",
  "Falta pouco para ficares em dia. Tu consegues!",
  "Não deixes para amanhã o que podes dar um passo hoje.",
];

const FRASES_NEUTRAS = [
  "Passo a passo chegamos lá: cada contribuição conta.",
  "Quem paga cedo descansa mais cedo.",
  "A tua contribuição faz a diferença. Obrigado!",
  "Um pequeno valor hoje evita a correria no último dia.",
  "Falta pouco para ficares em dia. Tu consegues!",
  "Não deixes para amanhã o que podes dar um passo hoje.",
];

export function fraseMotivacional(n: number, tomCristao = true): string {
  const lista = tomCristao ? FRASES_CRISTAS : FRASES_NEUTRAS;
  return lista[n % lista.length];
}

export function fraseParabens(tomCristao = true): string {
  return tomCristao ? "Que Deus abençoe a tua dedicação." : "Foi um prazer contar contigo.";
}

/**
 * Valor sugerido para hoje, calculado a partir do que falta e dos dias até ao prazo.
 * Encaixa na frase do modelo: "Hoje podes enviar {{sugestão}}."
 */
export function sugestaoDoDia(falta: number, dias: number, n: number): string {
  if (falta <= 0) return "";
  if (dias <= 0) return `${kz(falta)}, o valor em falta, para ficares em dia`;

  const ritmo = Math.min(falta, Math.max(100, Math.ceil(falta / dias / 100) * 100));
  if (ritmo >= falta) return `${kz(falta)} e ficas com tudo pago`;

  const dobro = Math.min(falta, ritmo * 2);
  switch (n % 3) {
    case 0:
      return kz(ritmo);
    case 1:
      return dobro >= falta ? `${kz(falta)} e ficas com tudo pago` : `${kz(dobro)}, se conseguires adiantar`;
    default:
      return `${kz(ritmo)}, ou o valor que estiver ao teu alcance`;
  }
}
