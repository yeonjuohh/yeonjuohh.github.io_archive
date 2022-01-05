/**
 * Bio component that queries for data
 * with Gatsby's useStaticQuery component
 *
 * See: https://www.gatsbyjs.com/docs/use-static-query/
 */

import * as React from "react"
import { Link } from "gatsby"
import { StaticImage } from "gatsby-plugin-image"

const Bio = () => {

  return (
    <div className="bio">
        <div className="bio_pic">
          <StaticImage
            className="bio-avatar"
            // layout="fixed"
            // formats={["auto", "webp", "avif"]}
            src="../images/profile-pic.png"
            width={70}
            height={70}
            // quality={100}
            alt="Profile picture"
          />
        </div>
        <div>
            <Link to="/about"><strong>Hado</strong></Link><br />
            세상에 관심이 많은 데이터 분석가입니다. <br />
        </div>
    </div>
  )
}

export default Bio
