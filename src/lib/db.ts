import { cert, getApps, initializeApp } from "firebase-admin/app";
import { Firestore, getFirestore, Timestamp, type DocumentSnapshot, type Query } from "firebase-admin/firestore";

let firestore: Firestore | null = null;

/** Ligação ao Firestore (só no servidor). Criada na primeira utilização. */
export function db(): Firestore {
  if (firestore) return firestore;
  const projectId = process.env.FIREBASE_PROJECT_ID;
  const clientEmail = process.env.FIREBASE_CLIENT_EMAIL;
  const privateKey = process.env.FIREBASE_PRIVATE_KEY;
  if (!projectId || !clientEmail || !privateKey) {
    throw new Error("Firebase não configurado: faltam FIREBASE_PROJECT_ID, FIREBASE_CLIENT_EMAIL ou FIREBASE_PRIVATE_KEY");
  }
  const app =
    getApps()[0] ??
    initializeApp({ credential: cert({ projectId, clientEmail, privateKey: privateKey.replace(/\\n/g, "\n") }) });
  const f = getFirestore(app);
  try {
    f.settings({ ignoreUndefinedProperties: true });
  } catch {
    // já inicializado (hot reload em desenvolvimento)
  }
  firestore = f;
  return f;
}

function converter(v: unknown): unknown {
  if (v instanceof Timestamp) return v.toDate();
  if (Array.isArray(v)) return v.map(converter);
  if (v && typeof v === "object") {
    return Object.fromEntries(Object.entries(v as Record<string, unknown>).map(([k, x]) => [k, converter(x)]));
  }
  return v;
}

/** Documento -> objeto com `id` e datas como Date. */
export function ler<T>(s: DocumentSnapshot): T | null {
  if (!s.exists) return null;
  return { ...(converter(s.data()) as object), id: s.id } as T;
}

export async function listar<T>(q: Query): Promise<T[]> {
  const r = await q.get();
  return r.docs.map((d) => ler<T>(d) as T);
}

/** Vários documentos de uma coleção por id, num Map. */
export async function porIds<T>(colecao: string, ids: string[]): Promise<Map<string, T>> {
  const unicos = [...new Set(ids)];
  const mapa = new Map<string, T>();
  for (let i = 0; i < unicos.length; i += 300) {
    const refs = unicos.slice(i, i + 300).map((id) => db().collection(colecao).doc(id));
    if (refs.length === 0) continue;
    const snaps = await db().getAll(...refs);
    for (const s of snaps) {
      const v = ler<T>(s);
      if (v) mapa.set(s.id, v);
    }
  }
  return mapa;
}
