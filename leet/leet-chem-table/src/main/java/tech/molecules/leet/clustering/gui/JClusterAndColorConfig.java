package tech.molecules.leet.clustering.gui;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.clustering.ClusterAppModel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class JClusterAndColorConfig extends JPanel {

    //private ClusterAppModel.Cluster cluster;
    private JLabel jl_cluster;
    private JLabel jl_color;
    private JTextField jtf;
    private JButton    jb_col;

    private boolean okPressed = false;

    public boolean wasOK() {
        return okPressed;
    }

    //Random r = new Random();
    //private Color color = Color.getHSBColor(r.nextFloat(),0.9f,0.8f);

    private Consumer<Pair<String,Color>> callback;

    public JClusterAndColorConfig(String name, Color col, Consumer<Pair<String,Color>> callback) {
        //this.cluster = c;
        this.callback = callback;
        reinit(name,col);
    }

    private JPanel getThisPanel() {return this;}

    public String getNewName() {
        return this.jtf.getText();
    }
    public Color getNewColor() {
        return this.jb_col.getBackground();
    }

    private void reinit(String name, Color col) {
        this.removeAll();
        this.setLayout(new BorderLayout());
        JPanel top = new JPanel();
        top.setLayout(new FlowLayout());
        this.jl_cluster = new JLabel("Cluster name: ");
        //this.jtf = new JTextField(this.cluster.getName(),20);
        this.jtf = new JTextField(name,20);
        this.jl_color = new JLabel("Color: ");
        this.jb_col = new JButton();
        this.jb_col.setPreferredSize(new Dimension(60,60));
        this.jb_col.setBackground(col);
        this.jb_col.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Color c_new = JColorChooser.showDialog(getThisPanel(),"Pick color",jtf.getBackground());
                //cluster.setColor(c_new);
                //color = c_new;
                //jb_col.setBackground(c_new);
                reinit(getNewName(),c_new);
            }
        });
        top.add(this.jl_cluster);
        top.add(this.jtf);
        top.add(jb_col);

        JPanel bottom = new JPanel();
        bottom.setLayout(new FlowLayout(FlowLayout.RIGHT));
        JButton jb_ok     = new JButton("OK");
        JButton jb_cancel = new JButton("Cancel");
        jb_ok.setEnabled(true);
        jb_cancel.setEnabled(true);
        bottom.add(jb_ok);
        bottom.add(jb_cancel);

        this.add(top,BorderLayout.CENTER);
        this.add(bottom,BorderLayout.SOUTH);

        jb_cancel.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                okPressed = false;
                //setVisible(false);
                //((JDialog)getParent()).dispose();
                callback.accept(null);
            }
        });
        jb_ok.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                okPressed = true;
                //((JDialog)getParent()).dispose();
                callback.accept(Pair.of(getNewName(),getNewColor()));
                //cluster.setName();
                //setVisible();
            }
        });
    }



    public static Pair<String, Color> showDialog(Component parent, ClusterAppModel.Cluster cluster) {
        JDialog di = new JDialog(SwingUtilities.getWindowAncestor(parent),"Define cluster", Dialog.ModalityType.APPLICATION_MODAL);
        di.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
        //ClusterAppModel.Cluster ci = m.Cluster("","",new ArrayList<>());
        List<Pair<String,Color>> newValues = new ArrayList<>();
        JClusterAndColorConfig pcluster = new JClusterAndColorConfig(cluster.getName(), cluster.getColor(), (x) -> {newValues.add(x); di.dispose(); } );
        di.setLocationRelativeTo(parent);
        di.setContentPane(pcluster);
        di.pack();
        di.setVisible(true);

        if(pcluster.wasOK()) {
            return Pair.of(pcluster.getNewName(),pcluster.getNewColor());
        }
        return null;
    }
}
