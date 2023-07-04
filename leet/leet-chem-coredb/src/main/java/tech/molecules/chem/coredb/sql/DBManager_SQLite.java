package tech.molecules.chem.coredb.sql;

import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;

public class DBManager_SQLite extends DBManager {


    public DBManager_SQLite(Connection connection) {
        super(connection, new SQLHelper.SqliteHelper());
    }

    public static void createDatabaseSchema_sqlite(Connection connection) throws SQLException {
        // TODO: add these two changes also to postgres init code..
        try (Statement statement = connection.createStatement()) {
            //statement.execute("CREATE TABLE IF NOT EXISTS data_type (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT)");
            statement.execute("CREATE TABLE IF NOT EXISTS project (id TEXT PRIMARY KEY, name TEXT)");
            statement.execute("CREATE TABLE IF NOT EXISTS assay (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, project_id TEXT, FOREIGN KEY(project_id) REFERENCES project(id))");
            //statement.execute("CREATE TABLE IF NOT EXISTS assay_parameter (id INTEGER PRIMARY KEY AUTOINCREMENT, assay_id INTEGER, data_type_id INTEGER, name TEXT, FOREIGN KEY(assay_id) REFERENCES assay(id), FOREIGN KEY(data_type_id) REFERENCES data_type(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS assay_parameter (id INTEGER PRIMARY KEY AUTOINCREMENT, assay_id INTEGER, data_type TEXT, name TEXT, FOREIGN KEY(assay_id) REFERENCES assay(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS compound (id TEXT PRIMARY KEY, idcode TEXT, idcode_coordinates TEXT)");
            statement.execute("CREATE TABLE IF NOT EXISTS batch (id TEXT PRIMARY KEY, compound_id TEXT, FOREIGN KEY(compound_id) REFERENCES compound(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS tube (id TEXT PRIMARY KEY, batch_id TEXT, FOREIGN KEY(batch_id) REFERENCES batch(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS assay_result (id INTEGER PRIMARY KEY AUTOINCREMENT, assay_id INTEGER, date DATE, tube_id TEXT, FOREIGN KEY(assay_id) REFERENCES assay(id), FOREIGN KEY(tube_id) REFERENCES tube(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS assay_result_data (assay_result_id INTEGER, assay_parameter_id INTEGER, double_value REAL, text_value TEXT, PRIMARY KEY (assay_result_id, assay_parameter_id), FOREIGN KEY(assay_result_id) REFERENCES assay_result(id), FOREIGN KEY(assay_parameter_id) REFERENCES assay_parameter(id))");
        }
    }

    @Override
    public void createDatabaseSchema() throws Exception {
        createDatabaseSchema_sqlite(getConnection());
    }

}
