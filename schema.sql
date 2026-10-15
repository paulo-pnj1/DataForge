-- DataForge (EDU + Business) - esquema Supabase
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


-- ── Organizações (escolas e empresas) e os seus membros ──────────────
-- Cada escola/empresa só vê os seus próprios dados. Uma pessoa (users) pode
-- pertencer a várias organizações com papéis diferentes.
create table if not exists organizacoes (
  id             text primary key,
  nome           text not null,
  tipo           text not null,                   -- escola | empresa
  produto        text not null,                   -- edu | business
  codigo_convite text not null unique,
  activa         boolean not null default true,
  created_at     timestamptz not null default now()
);

create table if not exists membros (
  org_id     text not null,
  username   text not null,
  papel      text not null,                       -- escola: professor | aluno | gestor; empresa: gestor | analista | membro
  created_at timestamptz not null default now(),
  primary key (org_id, username)
);
create index if not exists membros_user_idx on membros (username);

-- Turmas passam a pertencer a uma escola (opcional, para turmas já existentes).
alter table turmas add column if not exists org_id text;

-- ── Fase 2: convites, aprovação de empresas, pedidos de turma, emails ──
alter table users        add column if not exists must_change_pw boolean not null default false;
alter table turmas       add column if not exists aprovacao boolean not null default true;
alter table organizacoes add column if not exists estado text;       -- pendente | aprovada | recusada (vazio = aprovada)
alter table organizacoes add column if not exists dono   text;
alter table organizacoes add column if not exists motivo text;

create table if not exists convites_professor (
  codigo     text primary key,
  email      text not null,
  criado_por text not null,
  expira_em  timestamptz not null,
  usado_por  text,
  usado_em   timestamptz,
  cancelado  boolean not null default false,
  created_at timestamptz not null default now()
);

create table if not exists pedidos_turma (
  id         text primary key,                    -- codigo:username
  codigo     text not null,
  username   text not null,
  created_at timestamptz not null default now()
);

create table if not exists reset_tokens (
  token_hash text primary key,
  username   text not null,
  expira_em  timestamptz not null,
  usado      boolean not null default false,
  created_at timestamptz not null default now()
);

create table if not exists emails (
  id         text primary key,
  para       text not null,
  assunto    text not null,
  corpo      text,
  estado     text not null,                       -- enviado | fila | falhou
  erro       text,
  enviado_em timestamptz,
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
alter table organizacoes  enable row level security;
alter table membros       enable row level security;
alter table convites_professor enable row level security;
alter table pedidos_turma      enable row level security;
alter table reset_tokens       enable row level security;
alter table emails             enable row level security;
