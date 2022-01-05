import * as React from "react"
import Layout from "../components/layout"

const AboutPage = () => {
  return (
    <Layout location='/about' title='Hado Log'>
      <div ClassName="about">
      스타트업에서 데이터 분석가로 일하고 있습니다. <br /><br />

      👀   세상에 호기심이 많습니다. <br />
      🧳   여행을 좋아해서 틈만 나면 떠날 궁리를 합니다. <br />
      🦮   겁이 많고 예민한 강아지와 함께 살고 있습니다. <br />
      🙋‍   사회가 당연하다고 정한 것들에 딴지 거는걸 좋아합니다. <br />
      🌊   잘 하는 것 중에 재미있는 것을 찾기보다는 재미있는 것 중 잘하는 것을 선택하고자 합니다. <br /><br />

      서로의 경험을 공유할 수 있는 분들을 수집하고 있습니다. (줍줍) <br /><br />
      <a className="about-social" href="mailto:yeonju.oh5@gmail.com">
        ✉️   E-mail
      </a>
      </div>
    </Layout>
  );
};

export default AboutPage;