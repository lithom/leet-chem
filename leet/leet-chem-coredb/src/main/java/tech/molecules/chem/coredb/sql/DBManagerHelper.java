package tech.molecules.chem.coredb.sql;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class DBManagerHelper {



    public static DBManager_PostgreSQL getPostgres(String dbUrl, String username, String pw) throws SQLException {
        Connection connection = DriverManager.getConnection(dbUrl,username,pw);
        return new DBManager_PostgreSQL(connection);
    }

    public static DBManager_H2 getH2(String dbUrl) throws SQLException {
        Connection connection = DriverManager.getConnection(dbUrl);
        return new DBManager_H2(connection);
    }

    public static DBManager_SQLite getSQLite(String dbUrl) throws SQLException {
        Connection connection = DriverManager.getConnection(dbUrl);
        return new DBManager_SQLite(connection);
    }

}
