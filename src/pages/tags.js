import React from "react"
import PropTypes from "prop-types"
import Layout from "../components/layout"

import kebabCase from "lodash/kebabCase"

import { Helmet } from "react-helmet"
import { graphql, Link } from "gatsby"

const TagsPage = ({
  data: {
    allMarkdownRemark: { group },
    site: {
      siteMetadata: { title },
    },
  },
}) => {
    
return (
    
  <Layout location='/tags' title='Hado Log'>
  <div>
    <Helmet title={title} />
    <div>
      {/* <h2 className='tags-header'>Tags</h2> */}
      <div className='tags-context-container'>
        <ul>
            {group.map(tag => (
                <li key={tag.fieldValue}>
                    {/* <a href={'#' + kebabCase(tag.fieldValue)}> */}
                    {/* <a href={kebabCase(tag.fieldValue)}>                         */}
                    <Link to={`/tags/${kebabCase(tag.fieldValue)}/`}>
                        <div className='tags-context-list'>
                            <div className='blog-post-category'> {tag.fieldValue} </div>
                            <div> {tag.totalCount} </div>
                        </div>
                    {/* </a> */}
                    </Link>
                </li>
            ))}
        </ul>
        </div>        
    </div>

    {/* {group.map(tag => (
        <div className='tags-post-list'>
        <a name={kebabCase(tag.fieldValue)}>
            <h2>{tag.fieldValue}</h2>
        </a>
        {tag.title}
        </div>
    ))} */}

  </div>
  </Layout>
);
}

TagsPage.propTypes = {
    data: PropTypes.shape({
      allMarkdownRemark: PropTypes.shape({
        group: PropTypes.arrayOf(
          PropTypes.shape({
            fieldValue: PropTypes.string.isRequired,
            totalCount: PropTypes.number.isRequired,
          }).isRequired
        ),
      }),
      site: PropTypes.shape({
        siteMetadata: PropTypes.shape({
          title: PropTypes.string.isRequired,
        }),
      }),
    }),
  }
  export default TagsPage
  export const pageQuery = graphql`
    query {
      site {
        siteMetadata {
          title
        }
      }
      allMarkdownRemark(limit: 2000) {
        group(field: frontmatter___tags) {
          fieldValue
          totalCount
          edges {
              node {
                  fields {
                      slug
                  }
                  frontmatter {
                      title
                  }
              }
          }
        }
      }
    }
  `