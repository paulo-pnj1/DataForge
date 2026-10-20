import { v2 as cloudinary } from "cloudinary";

function configurar() {
  cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
    api_key: process.env.CLOUDINARY_API_KEY,
    api_secret: process.env.CLOUDINARY_API_SECRET,
    secure: true,
  });
}

export type Guardado = { publicId: string; tipo: string; formato: string | null };

/** Guarda o comprovativo como ficheiro "authenticated": só abre com URL assinada. Uma pasta por instituição. */
export function guardarComprovativo(buf: Buffer, nome: string, instituicaoId: string): Promise<Guardado> {
  configurar();
  return new Promise((resolve, reject) => {
    const stream = cloudinary.uploader.upload_stream(
      {
        folder: `muanapay/${instituicaoId}/comprovativos`,
        public_id: nome,
        resource_type: "auto",
        type: "authenticated",
        overwrite: false,
      },
      (erro, r) => {
        if (erro || !r) return reject(erro ?? new Error("Falha no upload para o Cloudinary"));
        resolve({ publicId: r.public_id, tipo: r.resource_type, formato: r.format ?? null });
      },
    );
    stream.end(buf);
  });
}

export function urlAssinada(publicId: string, tipo: string, formato: string | null): string {
  configurar();
  return cloudinary.url(publicId, {
    resource_type: tipo,
    type: "authenticated",
    sign_url: true,
    secure: true,
    ...(tipo === "image" && formato ? { format: formato } : {}),
  });
}

/** Apaga o ficheiro de um comprovativo. Se falhar só regista o erro: o registo na base de dados é o que importa. */
export async function apagarFicheiro(publicId: string, tipo: string): Promise<void> {
  configurar();
  try {
    await cloudinary.uploader.destroy(publicId, { resource_type: tipo, type: "authenticated", invalidate: true });
  } catch (e) {
    console.error("Cloudinary: não foi possível apagar", publicId, e);
  }
}
