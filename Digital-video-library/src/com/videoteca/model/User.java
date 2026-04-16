package com.videoteca.model;

/**
 * Base class for all user types.
 */
public abstract class User {
    private String userId;  
    private final String name;

    protected User(String userId, String name) {
        this.userId = userId;
        this.name = name;
    }

    /**
     * Constructor for new users before DB assignment.
     */
    protected User(String name) {
        this(null, name);
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public String getName() {
        return name;
    }

    /**
     * Behavior: whether the user may rent more items.
     */
    public abstract boolean canRentMore(int activeRentals);
}
