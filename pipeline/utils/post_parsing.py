"""Convert Mastodon status objects into pipeline post dictionaries."""

from html.parser import HTMLParser

from db import Field


class MastodonContentParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "p":
            self.parts.append("\n")

    def text(self):
        return " ".join("".join(self.parts).split())


def mastodon_content_to_text(content):
    parser = MastodonContentParser()
    parser.feed(content)
    return parser.text()


def post_to_pipeline_post(post):
    return {
        Field.ID: post[Field.ID],
        Field.TEXT: mastodon_content_to_text(post["content"]),
    }


def normalize_posts(posts):
    return [post_to_pipeline_post(post) for post in posts]
