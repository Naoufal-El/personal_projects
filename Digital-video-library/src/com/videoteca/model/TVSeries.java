package com.videoteca.model;

public class TVSeries extends MediaItem {
    private final int seasons;
    private final int episodes;

    public TVSeries(String id, String title, int year, double rating, int seasons, int episodes) {
        super(id, title, year, rating);
        this.seasons = seasons;
        this.episodes = episodes;
    }

    @Override
    public String getDetails() {
        return String.format(
            "TV Series: %s (%d) - %d seasons, %d episodes (Rating: %.1f)",
            getTitle(), getYear(), seasons, episodes, getRating()
        );
    }

    public int getSeasons() {
        return seasons;
    }

    public int getEpisodes() {
        return episodes;
    }
}