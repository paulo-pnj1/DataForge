import { loginAction } from "../actions";
import { Aviso, LogoGrande } from "@/components/ui";

export default async function Login({ searchParams }: { searchParams: Promise<{ erro?: string }> }) {
  const { erro } = await searchParams;
  return (
    <div className="login">
      <form action={loginAction}>
        <LogoGrande />
        {erro && <Aviso erro="Email ou senha incorretos." />}
        <div>
          <label htmlFor="email">Email</label>
          <input id="email" name="email" type="email" autoComplete="username" required />
        </div>
        <div>
          <label htmlFor="senha">Senha</label>
          <input id="senha" name="senha" type="password" autoComplete="current-password" required />
        </div>
        <button className="btn" type="submit">Entrar</button>
      </form>
    </div>
  );
}
