-- DataForge EDU - esquema Supabase
-- Cola este ficheiro no SQL Editor do Supabase e executa.

create table if not exists users (
  username      text primary key,
  name          text not null,
  email         text,
  password_hash text not null,
  role          text not null default 'aluno',   -- aluno | professor | admin
  turma         text,                            -- código da turma
  ranking       boolean not null default false,  -- aparece no ranking da turma?
  created_at    timestamptz not null default now()
);

create table if not exists progress (
  username   text primary key,
  data       jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

create table if not exists historico (
  id         bigserial primary key,
  username   text not null,
  data       jsonb not null,
  created_at timestamptz not null default now()
);
create index if not exists historico_user_idx on historico (username, created_at desc);

create table if not exists turmas (
  codigo     text primary key,
  nome       text not null,
  professor  text not null,
  created_at timestamptz not null default now()
);

create table if not exists certificados (
  codigo     text primary key,
  username   text not null,
  nome       text not null,
  tipo       text not null,
  emitido_em timestamptz not null default now()
);

create table if not exists erros (
  id         bigserial primary key,
  username   text,
  pagina     text,
  mensagem   text,
  created_at timestamptz not null default now()
);

-- Segurança: activa RLS SEM políticas. A chave pública (anon) fica sem acesso;
-- a app usa a chave service_role (guardada só em st.secrets, no servidor).
alter table users         enable row level security;
alter table progress      enable row level security;
alter table historico     enable row level security;
alter table turmas        enable row level security;
alter table certificados  enable row level security;
alter table erros         enable row level security;
