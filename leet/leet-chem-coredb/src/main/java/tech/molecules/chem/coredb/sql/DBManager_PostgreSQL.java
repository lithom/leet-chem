package tech.molecules.chem.coredb.sql;

import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;

public class DBManager_PostgreSQL extends DBManager {

    public DBManager_PostgreSQL(Connection connection) {
        super(connection, new SQLHelper.PostgresHelper());
    }

    public static void createDatabaseSchema_postgres(Connection connection) throws SQLException {
        try (Statement statement = connection.createStatement()) {
            statement.execute("CREATE TABLE IF NOT EXISTS data_type (id SERIAL PRIMARY KEY, name TEXT)");
            statement.execute("CREATE TABLE IF NOT EXISTS project (id TEXT PRIMARY KEY, name TEXT)");
            statement.execute("CREATE TABLE IF NOT EXISTS assay (id SERIAL PRIMARY KEY, name TEXT, project_id TEXT, FOREIGN KEY(project_id) REFERENCES project(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS assay_parameter (id SERIAL PRIMARY KEY, assay_id INTEGER, data_type_id INTEGER, name TEXT, FOREIGN KEY(assay_id) REFERENCES assay(id), FOREIGN KEY(data_type_id) REFERENCES data_type(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS compound (id TEXT PRIMARY KEY, idcode TEXT)");
            statement.execute("CREATE TABLE IF NOT EXISTS batch (id TEXT PRIMARY KEY, compound_id TEXT, FOREIGN KEY(compound_id) REFERENCES compound(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS tube (id TEXT PRIMARY KEY, batch_id TEXT, FOREIGN KEY(batch_id) REFERENCES batch(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS assay_result (id SERIAL PRIMARY KEY, assay_id INTEGER, date DATE, tube_id TEXT, FOREIGN KEY(assay_id) REFERENCES assay(id), FOREIGN KEY(tube_id) REFERENCES tube(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS assay_result_data (assay_result_id INTEGER, assay_parameter_id INTEGER, double_value DOUBLE PRECISION, text_value TEXT, PRIMARY KEY (assay_result_id, assay_parameter_id), FOREIGN KEY(assay_result_id) REFERENCES assay_result(id), FOREIGN KEY(assay_parameter_id) REFERENCES assay_parameter(id))");
        }
    }

    @Override
    public void createDatabaseSchema() throws Exception {
        createDatabaseSchema_postgres(getConnection());
    }

}
