package com.videoteca.persistence;

import com.videoteca.model.MediaItem;
import com.videoteca.model.Movie;
import com.videoteca.model.TVSeries;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public class MediaItemDAO {
    private final SQLiteDataSource ds;

    public MediaItemDAO(SQLiteDataSource ds) {
        this.ds = ds;
    }

    public List<MediaItem> findAll() throws SQLException {
        String sql = "SELECT * FROM media_items";
        try (Connection conn = ds.getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            List<MediaItem> list = new ArrayList<>();
            while (rs.next()) {
                String id = rs.getString("id");
                String title = rs.getString("title");
                int year = rs.getInt("year");
                double rating = rs.getDouble("rating");
                String type = rs.getString("type");
                if ("movie".equalsIgnoreCase(type)) {
                    int duration = rs.getInt("duration");
                    list.add(new Movie(id, title, year, rating, duration));
                } else {
                    int seasons = rs.getInt("seasons");
                    int episodes = rs.getInt("episodes");
                    list.add(new TVSeries(id, title, year, rating, seasons, episodes));
                }
            }
            return list;
        }
    }

    public MediaItem findById(String id) throws SQLException {
        String sql = "SELECT * FROM media_items WHERE id = ?";
        try (Connection conn = ds.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                if (rs.next()) {
                    String title = rs.getString("title");
                    int year = rs.getInt("year");
                    double rating = rs.getDouble("rating");
                    String type = rs.getString("type");
                    if ("movie".equalsIgnoreCase(type)) {
                        int duration = rs.getInt("duration");
                        return new Movie(id, title, year, rating, duration);
                    } else {
                        int seasons = rs.getInt("seasons");
                        int episodes = rs.getInt("episodes");
                        return new TVSeries(id, title, year, rating, seasons, episodes);
                    }
                }
                return null;
            }
        }
    }

    public void insert(MediaItem item) throws SQLException {
        String sql = "INSERT INTO media_items(id, title, year, rating, type, duration, seasons, episodes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)";
        try (Connection conn = ds.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, item.getId());
            ps.setString(2, item.getTitle());
            ps.setInt(3, item.getYear());
            ps.setDouble(4, item.getRating());
            if (item instanceof Movie) {
                Movie m = (Movie) item;
                ps.setString(5, "movie");
                ps.setInt(6, m.getDurationMinutes());
                ps.setNull(7, Types.INTEGER);
                ps.setNull(8, Types.INTEGER);
            } else {
                TVSeries s = (TVSeries) item;
                ps.setString(5, "tvseries");
                ps.setNull(6, Types.INTEGER);
                ps.setInt(7, s.getSeasons());
                ps.setInt(8, s.getEpisodes());
            }
            ps.executeUpdate();
        }
    }

    public void update(MediaItem item) throws SQLException {
        String sql = "UPDATE media_items SET title=?, year=?, rating=?, duration=?, seasons=?, episodes=? WHERE id=?";
        try (Connection conn = ds.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, item.getTitle());
            ps.setInt(2, item.getYear());
            ps.setDouble(3, item.getRating());
            if (item instanceof Movie) {
                Movie m = (Movie) item;
                ps.setInt(4, m.getDurationMinutes());
                ps.setNull(5, Types.INTEGER);
                ps.setNull(6, Types.INTEGER);
            } else {
                TVSeries s = (TVSeries) item;
                ps.setNull(4, Types.INTEGER);
                ps.setInt(5, s.getSeasons());
                ps.setInt(6, s.getEpisodes());
            }
            ps.setString(7, item.getId());
            ps.executeUpdate();
        }
    }

    public void delete(String id) throws SQLException {
        String sql = "DELETE FROM media_items WHERE id = ?";
        try (Connection conn = ds.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, id);
            ps.executeUpdate();
        }
    }
}