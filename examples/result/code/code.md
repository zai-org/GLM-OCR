relationship between the beans. The only difference from previous exercises is the change in the JNDI name element tag for the Address home interface:

```html

<local-jndi-name>AddressHomeLocal</local-jndi-name>

```

Because the Home interface for the Address is local, the tag is `<local-jndi-name>` rather than `<jndi-name>`.

The `weblogic-cmp-rdbms-jar.xml` descriptor file contains a number of new sections and elements in this exercise. A detailed examination of the relationship elements will wait until the next section.

The file contains a section mapping the Address <comp_fi1-1d> attributes from the `qb-jar.xml` file to the database columns in the `comp_fi1-1d` to a new section related to the primary key values in this

```html

<weblogic-rdbms-bean>

  <ejb-name>AddressJEJ</ejb-name>

  <data-source-name>titan-dataSource</data-source-name>

  <table-name>ADDRESS</table-name>

  <field-map>

    <cmp-field>id</cmp-field>

    <dbms-column>ID</dbms-column>

  </field-map>

  <field-map>

    <cmp-field>street</cmp-field>

    <dbms-column>STREET</dbms-column>

  </field-map>

  <field-map>

    <cmp-field>city</cmp-field>

    <dbms-column>CITY</dbms-column>

  </field-map>

  <field-map>

    <cmp-field>state</cmp-field>

    <dbms-column>STATE</dbms-column>

  </field-map>

  <field-map>

    <cmp-field>zip</cmp-field>

    <dbms-column>ZIP</dbms-column>

  </field-map>

  <!-- Automatically generate the value of ID in the database on

inserts using sequence table -->

  <automatic-key-generation>

    <generator-type>NAMED SEQUENCE TABLE</generator-type>

    <generator-name>ADDRESS SEQUENCE</generator-name>

    <key-cache-size>10</key-cache-size>

  </automatic-key-generation>

</weblogic-rdbms-bean>

```