"""The 17 Laws of the Game as published by the IFAB.

The HTML edition beats the PDF as an ingestion source: the clause hierarchy is
real <article> and <h2>/<h3> tags rather than font sizes and page coordinates,
and it carries amendment markup the PDF does not expose.
"""

from dataclasses import dataclass

BASE_URL = "https://www.theifab.com/laws/latest"


@dataclass(frozen=True)
class LawPage:
    number: int
    title: str
    slug: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.slug}/"


LAW_PAGES: tuple[LawPage, ...] = (
    LawPage(1, "The Field of Play", "the-field-of-play"),
    LawPage(2, "The Ball", "the-ball"),
    LawPage(3, "The Players", "the-players"),
    LawPage(4, "The Players' Equipment", "the-players-equipment"),
    LawPage(5, "The Referee", "the-referee"),
    LawPage(6, "The Other Match Officials", "the-other-match-officials"),
    LawPage(7, "The Duration of the Match", "the-duration-of-the-match"),
    LawPage(8, "The Start and Restart of Play", "the-start-and-restart-of-play"),
    LawPage(9, "The Ball in and out of Play", "the-ball-in-and-out-of-play"),
    LawPage(10, "Determining the Outcome of a Match", "determining-the-outcome-of-a-match"),
    LawPage(11, "Offside", "offside"),
    LawPage(12, "Fouls and Misconduct", "fouls-and-misconduct"),
    LawPage(13, "Free Kicks", "free-kicks"),
    LawPage(14, "The Penalty Kick", "the-penalty-kick"),
    LawPage(15, "The Throw-in", "the-throw-in"),
    LawPage(16, "The Goal Kick", "the-goal-kick"),
    LawPage(17, "The Corner Kick", "the-corner-kick"),
)
