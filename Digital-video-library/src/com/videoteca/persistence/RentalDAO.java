package com.videoteca.persistence;

import com.videoteca.model.Rental;

import java.sql.*;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

public class RentalDAO {
    private final SQLiteDataSource ds;

    public RentalDAO(SQLiteDataSource ds) {
        this.ds = ds;
    }

    public List<Rental> findAll() throws SQLException {
        String sql = "SELECT * FROM rentals";
        try (Connection conn = ds.getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            List<Rental> list = new ArrayList<>();
            while (rs.next()) {
                list.add(mapRow(rs));
            }
            return list;
        }
    }

    public Rental findById(String rentalId) throws SQLException {
        String sql = "SELECT * FROM rentals WHERE rentalId=?";
        try (Connection conn = ds.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, rentalId);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? mapRow(rs) : null;
            }
        }
    }

    private Rental mapRow(ResultSet rs) throws SQLException {
        String id = rs.getString("rentalId");
        String userId = rs.getString("userId");
        String mediaId = rs.getString("mediaId");
        LocalDate rented = LocalDate.parse(rs.getString("rentalDate"));
        Rental r = new Rental(id, userId, mediaId, rented);
        String ret = rs.getString("returnDate");
        if (ret != null) {
            r.setReturnDate(LocalDate.parse(ret));
        }
        return r;
    }

    public void insert(Rental r) throws SQLException {
        String sql = "INSERT INTO rentals(rentalId,userId,mediaId,rentalDate,returnDate) VALUES(?,?,?,?,?)";
        try (Connection conn = ds.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, r.getRentalId());
            ps.setString(2, r.getUserId());
            ps.setString(3, r.getMediaId());
            ps.setString(4, r.getRentalDate().toString());
            if (r.getReturnDate() != null) ps.setString(5, r.getReturnDate().toString()); else ps.setNull(5, Types.VARCHAR);
            ps.executeUpdate();
        }
    }

    public void update(Rental r) throws SQLException {
        String sql = "UPDATE rentals SET returnDate=? WHERE rentalId=?";
        try (Connection conn = ds.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            if (r.getReturnDate() != null) ps.setString(1, r.getReturnDate().toString()); else ps.setNull(1, Types.VARCHAR);
            ps.setString(2, r.getRentalId());
            ps.executeUpdate();
        }
    }

    public void delete(String id) throws SQLException {
        String sql = "DELETE FROM rentals WHERE rentalId=?";
        try (Connection conn = ds.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, id);
            ps.executeUpdate();
        }
    }
}