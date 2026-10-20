/** Aparece logo ao clicar numa ligação, enquanto a página nova carrega. */
export default function Carregando() {
  return (
    <div aria-busy="true" aria-label="A carregar">
      <div className="esqueleto" style={{ height: 34, width: "min(280px, 70%)", marginBottom: 12 }} />
      <div className="esqueleto" style={{ height: 16, width: "min(420px, 90%)", marginBottom: 28 }} />
      <div className="esqueleto" style={{ height: 120, marginBottom: 16 }} />
      <div className="esqueleto" style={{ height: 220 }} />
    </div>
  );
}
