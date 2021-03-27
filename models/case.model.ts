export interface Case {
    id?: number;
    classification?: Classification;
    score?: number;
    title?: string;
    description?: null | string;
    topic_id?: number;
    created_at?: Date;
    updated_at?: Date;
    privacy_related?: boolean | null;
    docbot_regex?: null | string;
    compiled_regex?: RegExp;
}

export enum Classification {
    Bad = "bad",
    Blocker = "blocker",
    Good = "good",
    Neutral = "neutral",
}
