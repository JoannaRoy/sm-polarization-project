export type Polarity = "agree" | "disagree";

export interface TopicSummary {
  id: string;
  label: string;
  sub_topic_count: number;
  argument_count: number;
  has_any_slates: boolean;
}

export interface Argument {
  id: string;
  text: string;
  post_id: string;
  topic_sentence: string | null;
  polarity: Polarity | null;
}

export interface RepresentativeStatement {
  id: string;
  round_index: number;
  statement: string;
  represented_ids: string[];
  represented_count: number;
}

export interface SubTopicDetail {
  id: string;
  label: string;
  polarity_target: string | null;
  count: number;
  arguments: Argument[];
  agree_slate: RepresentativeStatement[];
  disagree_slate: RepresentativeStatement[];
}

export interface Claim {
  id: string;
  text: string;
  topic_sentence: string | null;
  polarity: Polarity | null;
  sub_topic_id: string | null;
}

export interface TopicPost {
  id: string;
  text: string;
  claims: Claim[];
}

export interface TopicDetail {
  id: string;
  label: string;
  opinion_post_count: number;
  argument_count: number;
  posts: TopicPost[];
  sub_topics: SubTopicDetail[];
  total_sub_topic_count: number;
}

async function readError(res: Response): Promise<string> {
  const body = await res.json().catch(() => ({}) as { detail?: string });
  return body.detail ?? `${res.status} ${res.statusText}`;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await readError(res));
  return res.json() as Promise<T>;
}

export const fetchTopics = () => getJson<TopicSummary[]>("/topics");

export const fetchTopic = (id: string) =>
  getJson<TopicDetail>(`/topics/${encodeURIComponent(id)}`);

export interface TopicCandidate {
  id: string;
  label: string;
  score: number;
}

export async function matchPostTopics(
  content: string,
  k = 3,
): Promise<TopicCandidate[]> {
  const res = await fetch(`/match-post-topics?k=${k}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: `live-${Date.now()}`, content }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json() as Promise<TopicCandidate[]>;
}

export async function fetchTopicResponse(topicId: string): Promise<string> {
  const res = await fetch(`/topic-response/${encodeURIComponent(topicId)}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.text();
}
