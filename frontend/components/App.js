import React, { useState } from 'react';
import Map from './Map';
import Sidebar from './Sidebar';

const App = () => {
  const [selectedCluster, setSelectedCluster] = useState(null);

  return (
    <>
      <Map setSelectedCluster={setSelectedCluster} />
      <Sidebar selectedCluster={selectedCluster} />
    </>
  );
};

export default App;