/** @type {import('next').NextConfig} */
const config = {
  serverExternalPackages: ["firebase-admin"],
  poweredByHeader: false,
  async redirects() {
    // O termo "professor" deixou de existir no sistema: os endereços antigos continuam a funcionar.
    return [
      { source: "/professores", destination: "/participantes", permanent: true },
      { source: "/professores/:id", destination: "/participantes/:id", permanent: true },
    ];
  },
  async headers() {
    // Logotipos e ícones quase nunca mudam: o navegador guarda-os uma semana.
    return [
      {
        source: "/:all*(webp|png)",
        headers: [{ key: "Cache-Control", value: "public, max-age=604800, stale-while-revalidate=86400" }],
      },
    ];
  },
};
export default config;
