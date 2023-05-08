package tech.molecules.leet.datatable.swing2;

import javax.swing.*;
import java.awt.*;

public class InteractiveTable extends JPanel {

    private InteractiveTableModel model;

    private JScrollPane jsc = new JScrollPane();
    private InteractiveTablePanel itp;

    public InteractiveTable(InteractiveTableModel model) {
        this.model = model;
        reinit();
    }

    public void setModel(InteractiveTableModel model) {
        this.model = model;
        this.itp.setModel(model);
    }

    private void reinit() {
        this.removeAll();
        this.jsc = new JScrollPane();
        this.setLayout(new BorderLayout());;
        this.add(this.jsc,BorderLayout.CENTER);
        this.itp = new InteractiveTablePanel(model);
        this.jsc.setViewportView(this.itp);
        this.jsc.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
        this.jsc.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
    }

    public static void main(String args[]) {
        JFrame fi = new JFrame();
        fi.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        fi.setSize(600, 600);
        fi.getContentPane().setLayout(new BorderLayout());

        InteractiveTable it = new InteractiveTable( new InteractiveTableModel() );
        fi.getContentPane().add(it,BorderLayout.CENTER);
        fi.setVisible(true);

        Thread ri = new Thread() {
            @Override
            public void run() {
                while(true) {
                    try {
                        Thread.sleep(4000);
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                    //it.setModel(new InteractiveTableModel());
                }
            }
        };
        ri.start();
    }






}
