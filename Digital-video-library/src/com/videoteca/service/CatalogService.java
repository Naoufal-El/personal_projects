package com.videoteca.service;

import com.videoteca.model.MediaItem;
import com.videoteca.persistence.MediaItemDAO;
import com.videoteca.persistence.SQLiteDataSource;
import java.sql.SQLException;
import java.util.*;

public class CatalogService {
    private final MediaItemDAO mediaItemDAO;
    private final Map<String, MediaItem> items = new LinkedHashMap<>();

    public CatalogService(SQLiteDataSource ds) {
        this.mediaItemDAO = new MediaItemDAO(ds);
    }

    public void loadAll() throws SQLException {
        List<MediaItem> list = mediaItemDAO.findAll();
        items.clear();
        for (MediaItem m : list) {
            items.put(m.getId(), m);
        }
    }

    public void saveAll() throws SQLException {
        for (MediaItem m : items.values()) {
            if (mediaItemDAO.findById(m.getId()) != null) {
                mediaItemDAO.update(m);
            } else {
                mediaItemDAO.insert(m);
            }
        }
    }

    public List<MediaItem> listAll() {
        return Collections.unmodifiableList(new ArrayList<>(items.values()));
    }

    public MediaItem findById(String mediaId) {
        return items.get(mediaId);
    }

    public List<MediaItem> searchByTitle(String query) {
        List<MediaItem> results = new ArrayList<>();
        String q = query.toLowerCase();
        for (MediaItem m : items.values()) {
            if (m.getTitle().toLowerCase().contains(q)) {
                results.add(m);
            }
        }
        return results;
    }

    public void add(MediaItem m) throws SQLException {
        items.put(m.getId(), m);
        mediaItemDAO.insert(m);
    }

    public void remove(String mediaId) throws SQLException {
        items.remove(mediaId);
        mediaItemDAO.delete(mediaId);
    }
}
