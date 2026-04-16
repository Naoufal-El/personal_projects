package com.videoteca.model;

/**
 * Regular users may have up to 2 active rentals.
 */
public class RegularUser extends User {
    private static final int MAX_ACTIVE = 2;

    public RegularUser(String userId, String name) {
        super(userId, name);
    }

    public RegularUser(String name) {
        super(name);
    }

    @Override
    public boolean canRentMore(int activeRentals) {
        return activeRentals < MAX_ACTIVE;
    }
}
