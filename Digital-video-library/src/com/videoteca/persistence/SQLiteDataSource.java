package com.videoteca.persistence;

import com.videoteca.config.AppConfig;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

/**
 * Provides fresh SQLite connections configured with WAL and busy timeout.
 */
public class SQLiteDataSource {
    private final String url;

    public SQLiteDataSource() {
        this.url = AppConfig.get("db.url");
    }

    /**
     * Returns a new JDBC Connection with WAL and busy timeout enabled.
     */
    public Connection getConnection() {
        try {
            Class.forName("org.sqlite.JDBC");
            Connection conn = DriverManager.getConnection(url);
            try (Statement pragmas = conn.createStatement()) {
                pragmas.executeUpdate("PRAGMA journal_mode=WAL;");
                pragmas.executeUpdate("PRAGMA busy_timeout=5000;");
            }
            conn.setAutoCommit(true);
            return conn;
        } catch (ClassNotFoundException | SQLException e) {
            throw new RuntimeException("Failed to open SQLite connection", e);
        }
    }
}