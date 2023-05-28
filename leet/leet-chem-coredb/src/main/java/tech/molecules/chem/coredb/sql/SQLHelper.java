package tech.molecules.chem.coredb.sql;

public interface SQLHelper {
    public String getInsertOrIgnoreStatement(String tableName, String columns, String values);

    public class SqliteHelper implements SQLHelper {
        @Override
        public String getInsertOrIgnoreStatement(String tableName, String columns, String values) {
            return "INSERT OR IGNORE INTO " + tableName + " (" + columns + ") VALUES (" + values + ")";
        }
    }

    public class H2Helper implements SQLHelper {
        @Override
        public String getInsertOrIgnoreStatement(String tableName, String columns, String values) {
            return "MERGE INTO " + tableName + " (" + columns + ") KEY(" + columns + ") VALUES (" + values + ")";
        }
    }

    public class PostgresHelper implements SQLHelper {
        @Override
        public String getInsertOrIgnoreStatement(String tableName, String columns, String values) {
            return "INSERT INTO " + tableName + " (" + columns + ") VALUES (" + values + ") ON CONFLICT DO NOTHING";
        }
    }

}