-- Só as 2 tabelas novas do painel Business. Cola no Supabase > SQL Editor > Run.
-- (Podes correr várias vezes: não apaga nada.)

-- DataForge Business: dados e análises guardados por empresa (conteúdo comprimido em texto)
create table if not exists bn_datasets (
  id         text primary key,
  org_id     text not null,
  nome       text not null,
  conteudo   text not null,                       -- CSV comprimido (gzip + base64)
  linhas     integer not null default 0,
  colunas    integer not null default 0,
  reduzido   boolean not null default false,      -- true se foi guardada só uma amostra
  criado_por text not null,
  created_at timestamptz not null default now()
);
create index if not exists bn_datasets_org on bn_datasets (org_id);

create table if not exists bn_analises (
  id              text primary key,
  org_id          text not null,
  dataset_id      text,
  dataset_nome    text,
  criado_por      text not null,
  problema        text not null,                  -- vendas | stock | churn | fraude | segmentos
  titulo          text not null,
  alvo            text,
  resumo          jsonb,                          -- métricas, importâncias, avisos
  meta            jsonb,                          -- colunas usadas (para aplicar o modelo a dados novos)
  modelo          text,                           -- modelo treinado (joblib + base64), pode ser nulo
  modelo_guardado boolean not null default false,
  created_at      timestamptz not null default now()
);
create index if not exists bn_analises_org on bn_analises (org_id);

alter table bn_datasets enable row level security;
alter table bn_analises enable row level security;

-- Pede ao Supabase para reconhecer já as tabelas novas
notify pgrst, 'reload schema';
