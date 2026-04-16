package com.videoteca.persistence;

import com.videoteca.model.AdminUser;
import com.videoteca.model.PremiumUser;
import com.videoteca.model.RegularUser;
import com.videoteca.model.User;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public class UserDAO {
    private final SQLiteDataSource ds;

    public UserDAO(SQLiteDataSource ds) {
        this.ds = ds;
    }

    /**
     * Load all users from DB.
     */
    public List<User> findAll() throws SQLException {
        String sql = "SELECT userId, name, role FROM users";
        try (Connection conn = ds.getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {

            List<User> users = new ArrayList<>();
            while (rs.next()) {
                String id = rs.getString("userId");
                String name = rs.getString("name");
                String role = rs.getString("role").toLowerCase();
                switch (role) {
                    case "regular": users.add(new RegularUser(id, name)); break;
                    case "premium": users.add(new PremiumUser(id, name)); break;
                    case "admin":   users.add(new AdminUser(id, name));   break;
                    default: /* ignore */
                }
            }
            return users;
        }
    }

    /**
     * Find a single user by ID.
     */
    public User findById(String idStr) throws SQLException {
        String sql = "SELECT name, role FROM users WHERE userId = ?";
        try (Connection conn = ds.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            ps.setString(1, idStr);
            try (ResultSet rs = ps.executeQuery()) {
                if (!rs.next()) return null;
                String name = rs.getString("name");
                String role = rs.getString("role").toLowerCase();
                switch (role) {
                    case "regular": return new RegularUser(idStr, name);
                    case "premium": return new PremiumUser(idStr, name);
                    case "admin":   return new AdminUser(idStr, name);
                    default:        return null;
                }
            }
        }
    }

    /**
     * Insert a new user; returns the generated userId.
     */
    public String insert(User u) throws SQLException {
        String sql = "INSERT INTO users(userId, name, role) VALUES(?, ?, ?)";
        try (Connection conn = ds.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            // Generate new unique user ID with 'utente' prefix
            String newId;
            try (Statement idStmt = conn.createStatement();
                 ResultSet rs2 = idStmt.executeQuery("SELECT userId FROM users")) {
                int maxNum = 0;
                while (rs2.next()) {
                    String uid = rs2.getString(1);
                    if (uid != null && uid.startsWith("utente")) {
                        try {
                            int num = Integer.parseInt(uid.replaceAll("\\D", ""));
                            if (num > maxNum) {
                                maxNum = num;
                            }
                        } catch (NumberFormatException e) {
                            // ignore non-numeric IDs
                        }
                    }
                }
                newId = "utente" + (maxNum + 1);
            }

            ps.setString(1, newId);
            ps.setString(2, u.getName());
            String role = (u instanceof PremiumUser) ? "premium"
                        : (u instanceof AdminUser) ? "admin"
                        : "regular";
            ps.setString(3, role);
            ps.executeUpdate();
            u.setUserId(newId);
            return newId;
        }
    }

    /**
     * Update an existing user's name or role.
     */
    public void update(User u) throws SQLException {
        String sql = "UPDATE users SET name = ?, role = ? WHERE userId = ?";
        try (Connection conn = ds.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            ps.setString(1, u.getName());
            String role = (u instanceof PremiumUser) ? "premium"
                        : (u instanceof AdminUser) ? "admin"
                        : "regular";
            ps.setString(2, role);
            ps.setString(3, u.getUserId());
            ps.executeUpdate();
        }
    }

    /**
     * Delete user by ID.
     */
    public void delete(String id) throws SQLException {
        String sql = "DELETE FROM users WHERE userId = ?";
        try (Connection conn = ds.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, id);
            ps.executeUpdate();
        }
    }
}
