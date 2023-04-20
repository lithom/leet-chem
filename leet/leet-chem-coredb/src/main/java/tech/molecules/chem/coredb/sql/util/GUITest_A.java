package tech.molecules.chem.coredb.sql.util;

import tech.molecules.chem.coredb.gui.AssayTable;
import tech.molecules.chem.coredb.sql.DBManager;
import tech.molecules.chem.coredb.sql.SQLHelper;

import javax.swing.*;
import java.awt.*;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class GUITest_A {

    public static void main(String[] args) {
        long ts_a = System.currentTimeMillis();
        //ToyDataGenerator.generateToyData_A("sqlite");
        long ts_b = System.currentTimeMillis();
        DBManager dbManager = ToyDataGenerator.generateToyData_A("h2db_mem");
        long ts_c = System.currentTimeMillis();
        //System.out.println("sqlite:   "+(ts_b-ts_a));
        System.out.println("h2db_mem: "+(ts_c-ts_b));

        testGUI(dbManager);
//        String dbUrl = "jdbc:h2:mem:";
//        SQLHelper sqlhelper = new SQLHelper.H2Helper();
//
//        try (Connection connection = DriverManager.getConnection(dbUrl)) {
//            DBManager dbManager = new DBManager(connection, sqlhelper);
//            testGUI(dbManager);
//        } catch (SQLException e) {
//            throw new RuntimeException(e);
//        }

    }

    public static void testGUI(DBManager dbm) {
        JFrame fi = new JFrame();
        fi.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        fi.getContentPane().setLayout(new BorderLayout());

        AssayTable at = new AssayTable(dbm);
        fi.getContentPane().add(at,BorderLayout.CENTER);

        fi.setSize(600,600);
        fi.setVisible(true);
    }

}

